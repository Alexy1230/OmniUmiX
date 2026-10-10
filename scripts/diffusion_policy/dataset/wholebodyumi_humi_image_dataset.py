from typing import Dict
import copy
import glob
import os

import numpy as np
import torch
import zarr
from threadpoolctl import threadpool_limits

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.sampler import get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

register_codecs()


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, 1e-8)
    return vec / norm


def _ortho6d_to_rotation_matrix(ortho6d: np.ndarray) -> np.ndarray:
    x_raw = ortho6d[..., 0:3]
    y_raw = ortho6d[..., 3:6]
    x = _normalize_vector(x_raw)
    z = _normalize_vector(np.cross(x, y_raw))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=-1)


def _pose9d_to_hmat_batch(pose9d: np.ndarray) -> np.ndarray:
    pose9d = np.asarray(pose9d, dtype=np.float32)
    mats = np.tile(np.eye(4, dtype=np.float32), (pose9d.shape[0], 1, 1))
    mats[:, :3, 3] = pose9d[:, :3]
    mats[:, :3, :3] = _ortho6d_to_rotation_matrix(pose9d[:, 3:9])
    return mats


def _hmat_to_pose9d_batch(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=np.float32)
    pose9d = np.zeros((mats.shape[0], 9), dtype=np.float32)
    pose9d[:, :3] = mats[:, :3, 3]
    pose9d[:, 3:9] = mats[:, :3, :2].swapaxes(1, 2).reshape(mats.shape[0], 6)
    return pose9d


def _slice_stats(stat: Dict[str, np.ndarray], start: int, end: int) -> Dict[str, np.ndarray]:
    return {key: value[start:end] for key, value in stat.items()}


class _RunningStats:
    def __init__(self, dim: int):
        self.min = np.full((dim,), np.inf, dtype=np.float64)
        self.max = np.full((dim,), -np.inf, dtype=np.float64)
        self.sum = np.zeros((dim,), dtype=np.float64)
        self.sum_sq = np.zeros((dim,), dtype=np.float64)
        self.count = 0

    def update(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=np.float32).reshape(-1, self.min.shape[0])
        if value.size == 0:
            return
        self.min = np.minimum(self.min, value.min(axis=0))
        self.max = np.maximum(self.max, value.max(axis=0))
        self.sum += value.sum(axis=0)
        self.sum_sq += np.square(value.astype(np.float64)).sum(axis=0)
        self.count += value.shape[0]

    def finalize(self) -> Dict[str, np.ndarray]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize empty statistics.")
        mean = self.sum / self.count
        var = self.sum_sq / self.count - np.square(mean)
        std = np.sqrt(np.maximum(var, 0.0))
        return {
            "min": self.min.astype(np.float32),
            "max": self.max.astype(np.float32),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
        }


class WholeBodyUmiHumiImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        target_key: str = "target",
        proprio_key: str = "lower_body_dof",
        crop_size: int = 224,
        relative_action: bool = True,
        predict_next_step: bool = True,
        left_gripper_keypoint_index: int = 3,
        right_gripper_keypoint_index: int = 4,
        seed: int = 42,
        val_ratio: float = 0.05,
    ):
        self.shape_meta = shape_meta
        self.dataset_path = dataset_path
        self.target_key = target_key
        self.proprio_key = proprio_key
        self.crop_size = crop_size
        self.relative_action = relative_action
        self.predict_next_step = predict_next_step
        self.left_gripper_keypoint_index = left_gripper_keypoint_index
        self.right_gripper_keypoint_index = right_gripper_keypoint_index
        self.seed = seed
        self.val_ratio = val_ratio

        self._root = None
        self._store = None
        self._threadpool_limits_applied = False

        obs_meta = self.shape_meta["obs"]
        self.rgb_keys = [key for key, value in obs_meta.items() if value.get("type", "low_dim") == "rgb"]
        self.lowdim_keys = [key for key, value in obs_meta.items() if value.get("type", "low_dim") == "low_dim"]

        if sorted(self.rgb_keys) != ["left_fisheye_img", "right_fisheye_img"]:
            raise ValueError("Expected rgb obs keys to be left_fisheye_img and right_fisheye_img.")
        if self.lowdim_keys != [self.proprio_key]:
            raise ValueError(f"Expected low-dim obs key to be {self.proprio_key}.")

        self.img_obs_horizon = obs_meta["left_fisheye_img"]["horizon"]
        self.low_dim_obs_horizon = obs_meta[self.proprio_key]["horizon"]
        self.action_horizon = self.shape_meta["action"]["horizon"]

        root = self._get_root()
        data_group = root["data"]
        meta_group = root["meta"]

        if self.target_key not in data_group:
            raise KeyError(f"Missing target key {self.target_key} in dataset.")
        if self.proprio_key not in data_group:
            raise KeyError(f"Missing proprio key {self.proprio_key} in dataset.")
        for key in self.rgb_keys:
            if key not in data_group:
                raise KeyError(f"Missing image key {key} in dataset.")

        self.episode_ends = meta_group["episode_ends"][:].astype(np.int64)
        self.episode_starts = np.concatenate(([0], self.episode_ends[:-1]))
        self.n_episodes = len(self.episode_ends)

        target_dim = data_group[self.target_key].shape[-1]
        if target_dim < 11 or (target_dim - 2) % 9 != 0:
            raise ValueError(f"Unsupported target dimension {target_dim}.")
        self.num_keypoints = (target_dim - 2) // 9
        self.pose_dim = self.num_keypoints * 9
        self.left_gripper_index = self.pose_dim
        self.right_gripper_index = self.pose_dim + 1
        self.action_dim = self.num_keypoints * 10

        if self.shape_meta["action"]["shape"][0] != self.action_dim:
            raise ValueError(
                f"Configured action dim {self.shape_meta['action']['shape'][0]} does not match computed action dim {self.action_dim}."
            )
        if self.left_gripper_keypoint_index >= self.num_keypoints or self.right_gripper_keypoint_index >= self.num_keypoints:
            raise ValueError("Gripper keypoint index is out of range.")

        self.val_mask = get_val_mask(
            n_episodes=self.n_episodes,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
        self.episode_mask = ~self.val_mask
        self.sample_indices = self._build_sample_indices(self.episode_mask)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_root"] = None
        state["_store"] = None
        state["_threadpool_limits_applied"] = False
        return state

    def _resolve_dataset_path(self) -> str:
        dataset_path = os.path.expanduser(self.dataset_path)
        if dataset_path.endswith(".zarr.zip"):
            return dataset_path
        if os.path.isdir(dataset_path):
            zipped_candidates = sorted(glob.glob(os.path.join(dataset_path, "*.zarr.zip")))
            if len(zipped_candidates) == 1:
                return zipped_candidates[0]
            replay_buffer_path = os.path.join(dataset_path, "replay_buffer.zarr")
            if os.path.isdir(replay_buffer_path):
                return replay_buffer_path
        return dataset_path

    def _get_root(self):
        if self._root is None:
            resolved_path = self._resolve_dataset_path()
            if resolved_path.endswith(".zarr.zip"):
                self._store = zarr.ZipStore(resolved_path, mode="r")
                self._root = zarr.group(self._store)
            else:
                self._root = zarr.open(resolved_path, mode="r")
        return self._root

    def _build_sample_indices(self, episode_mask: np.ndarray) -> np.ndarray:
        indices = []
        for start, end, keep in zip(self.episode_starts, self.episode_ends, episode_mask):
            if not keep:
                continue
            indices.append(np.arange(start, end, dtype=np.int64))
        if len(indices) == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(indices, axis=0)

    def _get_episode_bounds(self, global_idx: int):
        episode_idx = int(np.searchsorted(self.episode_ends, global_idx, side="right"))
        return int(self.episode_starts[episode_idx]), int(self.episode_ends[episode_idx])

    def _get_history_indices(self, current_idx: int, episode_start: int, horizon: int) -> np.ndarray:
        offsets = np.arange(horizon - 1, -1, -1, dtype=np.int64)
        indices = current_idx - offsets
        return np.clip(indices, episode_start, current_idx)

    def _get_future_indices(self, current_idx: int, episode_end: int) -> np.ndarray:
        start = current_idx + 1 if self.predict_next_step else current_idx
        indices = start + np.arange(self.action_horizon, dtype=np.int64)
        return np.clip(indices, current_idx, episode_end - 1)

    def _center_crop(self, images: np.ndarray) -> np.ndarray:
        if images.ndim != 4:
            raise ValueError(f"Expected image batch with shape (T, H, W, C), got {images.shape}.")
        _, height, width, _ = images.shape
        if height < self.crop_size or width < self.crop_size:
            raise ValueError(f"Cannot crop images of shape {images.shape} to {self.crop_size}.")
        if height == self.crop_size and width == self.crop_size:
            return images
        top = (height - self.crop_size) // 2
        left = (width - self.crop_size) // 2
        return images[:, top:top + self.crop_size, left:left + self.crop_size, :]

    def _encode_action_chunk(self, reference_target: np.ndarray, future_target_chunk: np.ndarray) -> np.ndarray:
        future_target_chunk = np.asarray(future_target_chunk, dtype=np.float32)
        pose_chunk = future_target_chunk[:, :self.pose_dim].reshape(-1, self.num_keypoints, 9)

        if self.relative_action:
            reference_pose = reference_target[:self.pose_dim].reshape(self.num_keypoints, 9)
            encoded_pose_groups = []
            for keypoint_idx in range(self.num_keypoints):
                future_mats = _pose9d_to_hmat_batch(pose_chunk[:, keypoint_idx, :])
                base_mat = _pose9d_to_hmat_batch(reference_pose[keypoint_idx:keypoint_idx + 1])[0]
                relative_mats = np.einsum("ij,tjk->tik", np.linalg.inv(base_mat), future_mats)
                encoded_pose_groups.append(_hmat_to_pose9d_batch(relative_mats))
            pose_chunk = np.stack(encoded_pose_groups, axis=1)

        keypoint_widths = np.zeros((future_target_chunk.shape[0], self.num_keypoints, 1), dtype=np.float32)
        keypoint_widths[:, self.left_gripper_keypoint_index, 0] = future_target_chunk[:, self.left_gripper_index]
        keypoint_widths[:, self.right_gripper_keypoint_index, 0] = future_target_chunk[:, self.right_gripper_index]

        action_chunk = np.concatenate([pose_chunk, keypoint_widths], axis=-1)
        return action_chunk.reshape(future_target_chunk.shape[0], self.action_dim).astype(np.float32)

    def _get_action_chunk(self, global_idx: int) -> np.ndarray:
        root = self._get_root()
        target_arr = root["data"][self.target_key]
        _, episode_end = self._get_episode_bounds(global_idx)
        future_indices = self._get_future_indices(global_idx, episode_end)
        reference_target = target_arr[global_idx].astype(np.float32)
        future_target_chunk = target_arr[future_indices].astype(np.float32)
        return self._encode_action_chunk(reference_target, future_target_chunk)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.episode_mask = self.val_mask
        val_set.sample_indices = self._build_sample_indices(val_set.episode_mask)
        val_set._root = None
        val_set._store = None
        val_set._threadpool_limits_applied = False
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        root = self._get_root()
        action_stats = _RunningStats(self.action_dim)
        proprio_stats = _RunningStats(root["data"][self.proprio_key].shape[-1])

        for start, end, keep in zip(self.episode_starts, self.episode_ends, self.episode_mask):
            if not keep:
                continue

            proprio_stats.update(root["data"][self.proprio_key][start:end].astype(np.float32))
            target_episode = root["data"][self.target_key][start:end].astype(np.float32)
            episode_length = target_episode.shape[0]
            for step_idx in range(episode_length):
                global_idx = start + step_idx
                future_indices = self._get_future_indices(global_idx, end) - start
                future_target_chunk = target_episode[future_indices]
                action_chunk = self._encode_action_chunk(target_episode[step_idx], future_target_chunk)
                action_stats.update(action_chunk)

        normalizer = LinearNormalizer()
        action_stat = action_stats.finalize()
        action_normalizers = []
        for keypoint_idx in range(self.num_keypoints):
            group_start = keypoint_idx * 10
            group_stat = _slice_stats(action_stat, group_start, group_start + 10)
            action_normalizers.append(get_range_normalizer_from_stat(_slice_stats(group_stat, 0, 3)))
            action_normalizers.append(get_identity_normalizer_from_stat(_slice_stats(group_stat, 3, 9)))
            action_normalizers.append(get_range_normalizer_from_stat(_slice_stats(group_stat, 9, 10)))
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        normalizer[self.proprio_key] = get_range_normalizer_from_stat(proprio_stats.finalize())
        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        if len(self.sample_indices) == 0:
            return torch.zeros((0, self.action_horizon, self.action_dim), dtype=torch.float32)
        actions = [self._get_action_chunk(int(global_idx)) for global_idx in self.sample_indices]
        return torch.from_numpy(np.stack(actions, axis=0))

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self._threadpool_limits_applied:
            threadpool_limits(1)
            self._threadpool_limits_applied = True

        global_idx = int(self.sample_indices[idx])
        episode_start, episode_end = self._get_episode_bounds(global_idx)
        root = self._get_root()
        data_group = root["data"]

        image_indices = self._get_history_indices(global_idx, episode_start, self.img_obs_horizon)
        low_dim_indices = self._get_history_indices(global_idx, episode_start, self.low_dim_obs_horizon)

        obs_dict = {}
        for key in self.rgb_keys:
            image_batch = data_group[key][image_indices]
            image_batch = self._center_crop(image_batch)
            obs_dict[key] = np.moveaxis(image_batch, -1, 1).astype(np.float32) / 255.0

        obs_dict[self.proprio_key] = data_group[self.proprio_key][low_dim_indices].astype(np.float32)

        future_indices = self._get_future_indices(global_idx, episode_end)
        reference_target = data_group[self.target_key][global_idx].astype(np.float32)
        future_target_chunk = data_group[self.target_key][future_indices].astype(np.float32)
        action_chunk = self._encode_action_chunk(reference_target, future_target_chunk)

        return {
            "obs": {key: torch.from_numpy(value) for key, value in obs_dict.items()},
            "action": torch.from_numpy(action_chunk),
        }

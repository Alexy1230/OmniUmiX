"""Single-arm UMI observation history for pi0.5 (column-wise rotation 6D)."""

import dataclasses
from typing import Any

import numpy as np

from openpi import transforms
from openpi import transforms_umi as geom
from openpi.models import model as _model
from openpi.policies import umi_policy


def _check_pose(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.shape != shape or not np.isfinite(value).all():
        raise ValueError(f"{name} must be finite {shape}, got {value.shape}")
    first, second = value[..., 3:6], value[..., 6:9]
    if np.any(np.linalg.norm(first, axis=-1) < 1e-6) or np.any(
        np.linalg.norm(np.cross(first, second), axis=-1) < 1e-6
    ):
        raise ValueError(f"{name} has degenerate column-wise rotation 6D")
    return value


def make_state(history: Any, start: Any) -> np.ndarray:
    history = _check_pose(history, (2, 10), "tcp_pose9_gripper_history")
    start = _check_pose(start, (10,), "episode_start_tcp_pose9_gripper")
    poses = geom.pose9_to_matrix(history[:, :9])
    relative = geom.matrix_to_pose9(np.linalg.inv(poses[-1]) @ poses)
    start_relative = geom.matrix_to_pose9(np.linalg.inv(geom.pose9_to_matrix(start[:9])) @ poses)
    # Older frame first: relative xyz/rot6d, absolute gripper, start-relative rot6d.
    return np.concatenate([relative, history[:, 9:10], start_relative[:, 3:9]], axis=-1).reshape(32)


@dataclasses.dataclass(frozen=True)
class UmiProprioRepack(transforms.DataTransformFn):
    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        result = umi_policy.UmiRepack()(data)
        for key in ("tcp_pose9_gripper_history", "episode_start_tcp_pose9_gripper", "observation/image_previous"):
            result[key] = data[key]
        return result


@dataclasses.dataclass(frozen=True)
class UmiProprioInputs(transforms.DataTransformFn):
    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        state = make_state(data["tcp_pose9_gripper_history"], data["episode_start_tcp_pose9_gripper"])
        history = np.asarray(data["tcp_pose9_gripper_history"], dtype=np.float32)
        if "current_tcp_pose9_gripper" in data:
            reference = _check_pose(data["current_tcp_pose9_gripper"], (10,), "current_tcp_pose9_gripper")
            if not np.allclose(reference, history[-1], atol=1e-5, rtol=0):
                raise ValueError("Current TCP reference must equal the latest history frame")
        result = umi_policy.UmiInputs(self.model_type)(data)
        # Fixed temporal slots, both using the same wrist-image augmentation path.
        # These are two times from ONE camera, not two physical wrist cameras.
        previous_image = umi_policy._parse_image(data["observation/image_previous"])
        current_image = result["image"]["base_0_rgb"]
        if previous_image.shape != current_image.shape:
            raise ValueError("Previous/current images must have the same RGB shape")
        result["image"] = {
            "base_0_rgb": np.zeros_like(current_image),
            "left_wrist_0_rgb": previous_image,
            "right_wrist_0_rgb": current_image,
        }
        result["image_mask"] = {
            "base_0_rgb": np.False_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.True_,
        }
        result["state"] = state
        result["current_tcp_pose9_gripper"] = history[-1].copy()
        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32).copy()
            if actions.ndim != 2 or actions.shape[1] != 10:
                raise ValueError("Expected absolute [H,10] actions")
            _check_pose(actions, actions.shape, "actions")
            base = geom.pose9_to_matrix(history[-1, :9])
            actions[:, :9] = geom.matrix_to_pose9(np.linalg.inv(base) @ geom.pose9_to_matrix(actions[:, :9]))
            result["actions"] = actions
        return result


@dataclasses.dataclass(frozen=True)
class UmiProprioNormalize(transforms.DataTransformFn):
    norm_stats: dict[str, transforms.NormStats] | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.norm_stats:
            return data  # Explicitly skipping normalization, e.g. while computing statistics.
        for key, width in (("state", 32), ("actions", 10)):
            stats = self.norm_stats.get(key)
            if stats is None:
                raise ValueError(f"Missing {key} statistics: recompute stats with the proprio config")
            q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
            if q01.shape != (width,) or q99.shape != (width,):
                raise ValueError(f"Expected {width}-D {key} statistics for the proprio config")
            if not np.isfinite(q01).all() or not np.isfinite(q99).all() or np.any(q99 < q01):
                raise ValueError(f"Invalid {key} quantile statistics")
        data = geom.UMIAction10Normalize(self.norm_stats)(data)
        return transforms.Normalize({"state": self.norm_stats["state"]}, use_quantiles=True)(data)


class UmiHistoryDataset:
    """Fetch paired images/states at the same two indices within an episode."""

    def __init__(self, dataset: Any, history_steps: int):
        if not isinstance(history_steps, int) or history_steps < 1:
            raise ValueError("UMI history steps must be a positive integer")
        self.dataset = dataset
        self.history_steps = history_steps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = dict(self.dataset[index])
        current = self.dataset.hf_dataset[index]
        start_index = int(index) - int(current["frame_index"])
        if start_index < 0:
            raise ValueError("Invalid episode/frame indices")
        start = self.dataset.hf_dataset[start_index]
        previous_index = max(start_index, int(index) - self.history_steps)
        previous = self.dataset.hf_dataset[previous_index]
        if any(int(row["episode_index"]) != int(current["episode_index"]) for row in (start, previous)) or int(
            start["frame_index"]
        ) != 0:
            raise ValueError("History lookup crossed episode boundary")
        # Read through LeRobot so video decoding uses this row's timestamp too.
        # At the episode start both times coincide; reuse the decoded current image.
        previous_sample = sample if previous_index == int(index) else self.dataset[previous_index]
        sample["observation/image_previous"] = previous_sample["observation.images.fisheye_img"]
        sample["tcp_pose9_gripper_history"] = np.stack(
            [previous["observation.state"], current["observation.state"]]
        )
        sample["episode_start_tcp_pose9_gripper"] = np.asarray(start["observation.state"])
        return sample

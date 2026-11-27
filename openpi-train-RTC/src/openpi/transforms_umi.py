"""Transforms for UMI TCP pose9 + gripper actions.

UMI actions are represented as ``[x, y, z, rot6d, gripper]`` where the
rotation is stored as the first two columns of a rotation matrix, flattened
column-wise.  The model is trained on relative TCP actions whose reference is
the first action in each action chunk.
"""

import dataclasses
from typing import Any

import numpy as np

from openpi import transforms

UMI_ACTION_DIM = 10
UMI_POSE_DIM = 9
UMI_GRIPPER_INDEX = 9
_POSITION_DIMS = (0, 1, 2)
_NORMALIZED_DIMS = (0, 1, 2, 9)


def _check_action_shape(actions: np.ndarray) -> None:
    if actions.ndim < 2 or actions.shape[-1] < UMI_ACTION_DIM:
        raise ValueError(f"Expected actions with shape [..., horizon, >=10], got {actions.shape}.")


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert UMI column-wise rot6d to rotation matrices.

    ``rot6d`` has shape ``[..., 6]`` and is interpreted as
    ``[R[:, 0], R[:, 1]]``.  The input is orthogonalized so slightly noisy
    dataset values still produce a valid SO(3) matrix.
    """

    first = rot6d[..., :3]
    second = rot6d[..., 3:6]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    first_unit = first / np.maximum(first_norm, 1e-8)
    second_orthogonal = second - np.sum(first_unit * second, axis=-1, keepdims=True) * first_unit
    second_norm = np.linalg.norm(second_orthogonal, axis=-1, keepdims=True)
    second_unit = second_orthogonal / np.maximum(second_norm, 1e-8)
    third_unit = np.cross(first_unit, second_unit, axis=-1)
    return np.stack([first_unit, second_unit, third_unit], axis=-1)


def _matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to UMI's column-wise rot6d representation."""

    return np.stack([matrix[..., :, 0], matrix[..., :, 1]], axis=-2).reshape(*matrix.shape[:-2], 6)


def pose9_to_matrix(pose9: np.ndarray) -> np.ndarray:
    pose9 = np.asarray(pose9, dtype=np.float32)
    if pose9.shape[-1] < UMI_POSE_DIM:
        raise ValueError(f"Expected pose9 with last dimension >=9, got {pose9.shape}.")
    matrix = np.zeros((*pose9.shape[:-1], 4, 4), dtype=pose9.dtype)
    matrix[..., :3, :3] = _rot6d_to_matrix(pose9[..., 3:9])
    matrix[..., :3, 3] = pose9[..., :3]
    matrix[..., 3, 3] = 1.0
    return matrix


def matrix_to_pose9(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape[-2:] != (4, 4):
        raise ValueError(f"Expected homogeneous matrices with shape [...,4,4], got {matrix.shape}.")
    return np.concatenate(
        [matrix[..., :3, 3], _matrix_to_rot6d(matrix[..., :3, :3])],
        axis=-1,
    )


def relative_pose9_from_first(pose9: np.ndarray) -> np.ndarray:
    """Express every pose in a chunk in the coordinate frame of its first pose."""

    pose9 = np.asarray(pose9, dtype=np.float32)
    if pose9.ndim < 2 or pose9.shape[-1] < UMI_POSE_DIM:
        raise ValueError(f"Expected pose sequence with shape [..., horizon, >=9], got {pose9.shape}.")
    matrices = pose9_to_matrix(pose9[..., :UMI_POSE_DIM])
    base = matrices[..., :1, :, :]
    relative = np.linalg.inv(base) @ matrices
    return matrix_to_pose9(relative)


def absolute_pose9_from_first(relative_pose9: np.ndarray, base_pose9: np.ndarray) -> np.ndarray:
    """Recover absolute poses from relative poses and a base pose."""

    relative_pose9 = np.asarray(relative_pose9, dtype=np.float32)
    base_pose9 = np.asarray(base_pose9, dtype=np.float32)
    if relative_pose9.ndim < 2 or relative_pose9.shape[-1] < UMI_POSE_DIM:
        raise ValueError(f"Expected relative poses with shape [..., horizon, >=9], got {relative_pose9.shape}.")
    if base_pose9.shape[-1] < UMI_POSE_DIM:
        raise ValueError(f"Expected base pose with last dimension >=9, got {base_pose9.shape}.")
    base_matrix = pose9_to_matrix(base_pose9[..., :UMI_POSE_DIM])
    relative_matrix = pose9_to_matrix(relative_pose9[..., :UMI_POSE_DIM])
    absolute = base_matrix[..., None, :, :] @ relative_matrix
    return matrix_to_pose9(absolute)


@dataclasses.dataclass(frozen=True)
class RelativeTcpPose9ColumnFromFirstAction(transforms.DataTransformFn):
    """Convert absolute UMI action chunks to first-action-relative chunks."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)
        _check_action_shape(actions)
        result = actions.copy()
        result[..., :UMI_POSE_DIM] = relative_pose9_from_first(actions[..., :UMI_POSE_DIM])
        # Gripper is an absolute scalar and is deliberately not transformed.
        result[..., UMI_GRIPPER_INDEX] = actions[..., UMI_GRIPPER_INDEX]
        data["actions"] = result
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteTcpPose9ColumnFromFirstAction(transforms.DataTransformFn):
    """Convert model-relative UMI actions to absolute actions using a deployment base."""

    reference_key: str = "current_tcp_pose9_gripper"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data
        if self.reference_key not in data:
            raise ValueError(f"Missing deployment reference {self.reference_key!r}.")
        actions = np.asarray(data["actions"], dtype=np.float32)
        reference = np.asarray(data[self.reference_key], dtype=np.float32)
        _check_action_shape(actions)
        relative = actions[..., :UMI_ACTION_DIM] if actions.shape[-1] > UMI_ACTION_DIM else actions
        result = np.array(relative, copy=True)
        result[..., :UMI_POSE_DIM] = absolute_pose9_from_first(relative[..., :UMI_POSE_DIM], reference[..., :UMI_POSE_DIM])
        # Gripper is absolute and must be copied, not composed with the base.
        result[..., UMI_GRIPPER_INDEX] = relative[..., UMI_GRIPPER_INDEX]
        data["actions"] = result
        return data


@dataclasses.dataclass(frozen=True)
class UMIAction10Normalize(transforms.DataTransformFn):
    """Normalize UMI relative xyz/gripper while preserving rot6d values."""

    norm_stats: dict[str, transforms.NormStats] | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.norm_stats is None or "actions" not in data:
            return data
        stats = self.norm_stats.get("actions")
        if stats is None:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32).copy()
        _check_action_shape(actions)
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("UMI action normalization requires q01 and q99 action statistics.")
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        # normalize to [-1, 1]
        for dim in _NORMALIZED_DIMS:
            actions[..., dim] = (actions[..., dim] - q01[dim]) / (q99[dim] - q01[dim] + 1e-6) * 2.0 - 1.0
        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class UMIAction10Unnormalize(transforms.DataTransformFn):
    """Undo :class:`UMIAction10Normalize` on the first ten action dimensions."""

    norm_stats: dict[str, transforms.NormStats] | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.norm_stats is None or "actions" not in data:
            return data
        stats = self.norm_stats.get("actions")
        if stats is None:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32).copy()
        _check_action_shape(actions)
        if stats.q01 is None or stats.q99 is None:
            raise ValueError("UMI action unnormalization requires q01 and q99 action statistics.")
        q01 = np.asarray(stats.q01, dtype=np.float32)
        q99 = np.asarray(stats.q99, dtype=np.float32)
        for dim in _NORMALIZED_DIMS:
            actions[..., dim] = (actions[..., dim] + 1.0) / 2.0 * (q99[dim] - q01[dim] + 1e-6) + q01[dim]
        data["actions"] = actions
        return data

"""Right-first bimanual UMI transforms, reusing single-arm pose geometry."""

import dataclasses
from typing import Any

import numpy as np

from openpi import transforms
from openpi import transforms_umi

UMI_BIMANUAL_ACTION_DIM = 20
_NORMALIZED_DIMS = (0, 1, 2, 9, 10, 11, 12, 19)


def _check_action_shape(actions: np.ndarray) -> None:
    if actions.ndim < 2 or actions.shape[-2] == 0 or actions.shape[-1] < UMI_BIMANUAL_ACTION_DIM:
        raise ValueError(f"Expected actions [..., horizon, >=20], got {actions.shape}.")
    if not np.isfinite(actions[..., :UMI_BIMANUAL_ACTION_DIM]).all():
        raise ValueError("Bimanual actions must be finite.")


@dataclasses.dataclass(frozen=True)
class RelativeBimanualTcpPose9ColumnFromFirstAction(transforms.DataTransformFn):
    """Express each arm's chunk relative to that arm's first action."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data
        actions = np.asarray(data["actions"], dtype=np.float32)
        _check_action_shape(actions)
        result = actions.copy()
        for start in (0, 10):
            result[..., start:start + 9] = transforms_umi.relative_pose9_from_first(actions[..., start:start + 9])
        # Both grippers remain absolute scalars.
        data["actions"] = result
        return data


@dataclasses.dataclass(frozen=True)
class AbsoluteBimanualTcpPose9ColumnFromFirstAction(transforms.DataTransformFn):
    """Anchor each arm's predictions to its own deployment TCP reference."""

    reference_key: str = "current_tcp_pose9_gripper"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "actions" not in data:
            return data
        if self.reference_key not in data:
            raise ValueError(f"Missing deployment reference {self.reference_key!r}.")
        actions = np.asarray(data["actions"], dtype=np.float32)
        reference = np.asarray(data[self.reference_key], dtype=np.float32)
        _check_action_shape(actions)
        if reference.shape != actions.shape[:-2] + (20,) or not np.isfinite(reference).all():
            raise ValueError(f"Expected reference {actions.shape[:-2] + (20,)}, got {reference.shape}.")
        result = actions[..., :20].copy()
        for start in (0, 10):
            result[..., start:start + 9] = transforms_umi.absolute_pose9_from_first(
                actions[..., start:start + 9], reference[..., start:start + 9]
            )
        data["actions"] = result
        return data


def _normalize_actions(
    data: dict[str, Any], norm_stats: dict[str, transforms.NormStats] | None, *, inverse: bool
) -> dict[str, Any]:
    if norm_stats is None or "actions" not in data or "actions" not in norm_stats:
        return data
    stats = norm_stats["actions"]
    actions = np.asarray(data["actions"], dtype=np.float32).copy()
    _check_action_shape(actions)
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Bimanual UMI normalization requires q01 and q99 action statistics.")
    q01, q99 = np.asarray(stats.q01), np.asarray(stats.q99)
    if q01.shape != (20,) or q99.shape != (20,):
        raise ValueError("Bimanual UMI requires new 20-D action statistics; single-arm stats cannot be reused.")
    for dim in _NORMALIZED_DIMS:
        scale = q99[dim] - q01[dim] + 1e-6
        if inverse:
            actions[..., dim] = (actions[..., dim] + 1.0) / 2.0 * scale + q01[dim]
        else:
            actions[..., dim] = (actions[..., dim] - q01[dim]) / scale * 2.0 - 1.0
    data["actions"] = actions
    return data


@dataclasses.dataclass(frozen=True)
class UMIAction20Normalize(transforms.DataTransformFn):
    """Normalize xyz/gripper for both arms, preserving rotations and padding."""

    norm_stats: dict[str, transforms.NormStats] | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return _normalize_actions(data, self.norm_stats, inverse=False)


@dataclasses.dataclass(frozen=True)
class UMIAction20Unnormalize(transforms.DataTransformFn):
    """Undo UMIAction20Normalize before converting both poses to absolute."""

    norm_stats: dict[str, transforms.NormStats] | None

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        return _normalize_actions(data, self.norm_stats, inverse=True)

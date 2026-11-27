"""UMI policy input/output adapters for openpi."""

import dataclasses
from typing import Any

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_umi_example() -> dict[str, Any]:
    return {
        "observation/image": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "current_tcp_pose9_gripper": np.zeros(10, dtype=np.float32),
        "prompt": "do something",
    }


def _parse_image(image: Any) -> np.ndarray:
    """
    Return an HWC uint8 RGB image
    """
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError(f"Expected a single HWC or CHW image, got {image.shape}.")
    if image.shape[0] == 3 and image.shape[-1] != 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {image.shape}.")
    if np.issubdtype(image.dtype, np.floating):
        # LeRobot image tensors are usually in [0, 1]. Also accept already
        # scaled [0, 255] floating point arrays.
        scale = 255.0 if np.nanmax(image) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return image


@dataclasses.dataclass(frozen=True)
class UmiRepack(transforms.DataTransformFn):
    """Map both LeRobot storage keys and deployment keys to UMI policy keys."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if "observation.images.fisheye_img" in data:
            image = data["observation.images.fisheye_img"]  # LeRobot
        elif "observation/image" in data:
            image = data["observation/image"]
        else:
            raise KeyError("UMI input is missing observation.images.fisheye_img or observation/image.")

        result: dict[str, Any] = {"observation/image": image}  # UMI sample format
        if "action" in data:
            result["actions"] = data["action"]
        elif "actions" in data:
            result["actions"] = data["actions"]
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        if "current_tcp_pose9_gripper" in data:
            result["current_tcp_pose9_gripper"] = data["current_tcp_pose9_gripper"]
        return result


@dataclasses.dataclass(frozen=True)
class UmiInputs(transforms.DataTransformFn):
    """Convert a UMI sample to openpi's image/state/action dictionary."""

    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        base_image = _parse_image(data["observation/image"])
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"UMI currently supports pi0/pi0.5, got {self.model_type}.")

        inputs: dict[str, Any] = {
            "state": np.zeros((0,), dtype=np.float32),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "current_tcp_pose9_gripper" in data:
            reference = np.asarray(data["current_tcp_pose9_gripper"], dtype=np.float32)
            if reference.shape[-1] != 10:
                raise ValueError(f"Expected current_tcp_pose9_gripper shape [...,10], got {reference.shape}.")
            inputs["current_tcp_pose9_gripper"] = reference
        return inputs


@dataclasses.dataclass(frozen=True)
class UmiOutputs(transforms.DataTransformFn):
    """Convert relative model actions to absolute 32-step UMI actions."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] != 32 or actions.shape[1] < 10:
            raise ValueError(f"Expected model output shape [32,>=10], got {actions.shape}.")
        if "current_tcp_pose9_gripper" not in data:
            raise ValueError("UMI absolute output requires current_tcp_pose9_gripper.")
        return {"actions": actions[..., :10]}

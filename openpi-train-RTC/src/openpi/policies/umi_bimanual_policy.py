"""Bimanual UMI adapters: right arm first, left arm second."""

import dataclasses
from typing import Any

import numpy as np

from openpi import transforms
from openpi.models import model as _model
from openpi.policies import umi_policy


def make_umi_bimanual_example() -> dict[str, Any]:
    pose = [0, 0, 0, 1, 0, 0, 0, 1, 0, 0]
    return {
        "observation/image_right": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "observation/image_left": np.random.randint(256, size=(240, 240, 3), dtype=np.uint8),
        "current_tcp_pose9_gripper": np.array(pose * 2, dtype=np.float32),
        "prompt": "do something with both hands",
    }


@dataclasses.dataclass(frozen=True)
class UmiBimanualRepack(transforms.DataTransformFn):
    """Map two LeRobot image keys and a right-first 20-D action to policy keys."""

    right_image_key: str = "observation.images.fisheye_img_right"
    left_image_key: str = "observation.images.fisheye_img_left"

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.right_image_key == self.left_image_key:
            raise ValueError("Bimanual UMI requires distinct right and left image keys.")
        
        result: dict[str, Any] = {}
        for dataset_key, policy_key in (
            (self.right_image_key, "observation/image_right"),
            (self.left_image_key, "observation/image_left"),
        ):
            if dataset_key in data:
                result[policy_key] = data[dataset_key]
            elif policy_key in data:
                result[policy_key] = data[policy_key]
            else:
                raise KeyError(f"Bimanual UMI input is missing {dataset_key} or {policy_key}.")
            
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
class UmiBimanualInputs(transforms.DataTransformFn):
    """Use two wrist images and an empty semantic state, as in single-arm UMI."""

    model_type: _model.ModelType

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        if self.model_type not in (_model.ModelType.PI0, _model.ModelType.PI05):
            raise ValueError(f"Bimanual UMI supports pi0/pi0.5, got {self.model_type}.")
        right_image = umi_policy._parse_image(data["observation/image_right"])
        left_image = umi_policy._parse_image(data["observation/image_left"])
        inputs: dict[str, Any] = {
            "state": np.zeros((0,), dtype=np.float32),
            "image": {
                "base_0_rgb": np.zeros_like(right_image),
                "left_wrist_0_rgb": left_image,
                "right_wrist_0_rgb": right_image,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[-1] != 20 or not np.isfinite(actions).all():
                raise ValueError(f"Expected finite bimanual actions [T,20], got {actions.shape}.")
            inputs["actions"] = actions
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "current_tcp_pose9_gripper" in data:
            reference = np.asarray(data["current_tcp_pose9_gripper"], dtype=np.float32)
            if reference.shape != (20,) or not np.isfinite(reference).all():
                raise ValueError(f"Expected finite current_tcp_pose9_gripper [20], got {reference.shape}.")
            inputs["current_tcp_pose9_gripper"] = reference
        return inputs


@dataclasses.dataclass(frozen=True)
class UmiBimanualOutputs(transforms.DataTransformFn):
    """Return 32 absolute targets per arm after the output pose transform."""

    def __call__(self, data: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] != 32 or actions.shape[1] < 20:
            raise ValueError(f"Expected model output [32,>=20], got {actions.shape}.")
        if "current_tcp_pose9_gripper" not in data:
            raise ValueError("Bimanual UMI absolute output requires current_tcp_pose9_gripper.")
        if not np.isfinite(actions[..., :20]).all():
            raise ValueError("Bimanual UMI output contains non-finite actions.")
        return {"actions": actions[..., :20]}

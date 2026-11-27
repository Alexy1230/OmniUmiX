import numpy as np

from openpi.models import model
from openpi.policies import umi_policy


def test_umi_repack_drops_raw_state_and_maps_action():
    """
    Test LeRobot -> UMI sample format
    """
    raw = {
        "observation.images.fisheye_img": np.zeros((3, 240, 240), dtype=np.float32),
        "observation.state": np.ones(10, dtype=np.float32),
        "action": np.zeros((32, 10), dtype=np.float32),
        "prompt": "lift the bottle",
    }
    result = umi_policy.UmiRepack()(raw)
    assert "observation.state" not in result
    assert "state" not in result
    assert result["actions"].shape == (32, 10)


def test_umi_inputs_use_empty_state_and_one_camera():
    """
    Test UMI sample format -> deployment (Pi Model required format)
    """
    data = {
        "observation/image": np.zeros((3, 240, 240), dtype=np.float32),
        "actions": np.zeros((32, 10), dtype=np.float32),
        "prompt": "lift the bottle",
    }
    result = umi_policy.UmiInputs(model_type=model.ModelType.PI05)(data)
    assert result["state"].shape == (0,)
    assert result["image"]["base_0_rgb"].shape == (240, 240, 3)
    assert bool(result["image_mask"]["base_0_rgb"])
    assert not bool(result["image_mask"]["left_wrist_0_rgb"])
    assert result["actions"].shape == (32, 10)


def test_umi_outputs_requires_reference_and_returns_ten_dims():
    data = {
        "actions": np.zeros((32, 32), dtype=np.float32),
        "current_tcp_pose9_gripper": np.zeros(10, dtype=np.float32),
    }
    result = umi_policy.UmiOutputs()(data)
    assert result["actions"].shape == (32, 10)

import numpy as np
import pytest

from openpi.models import model
from openpi.policies import umi_bimanual_policy as policy
from openpi import transforms_umi_bimanual as transforms
from openpi.shared import normalize


def test_repack_two_images_drops_measured_state_and_maps_actions():
    raw = {
        "observation.images.fisheye_img_right": np.full((3, 240, 240), .25, np.float32),
        "observation.images.fisheye_img_left": np.full((3, 240, 240), .75, np.float32),
        "observation.state": np.ones(20), "action": np.zeros((32, 20)), "prompt": "fold",
    }
    repacked = policy.UmiBimanualRepack()(raw)
    data = policy.UmiBimanualInputs(model.ModelType.PI05)(repacked)
    assert data["state"].shape == (0,)
    assert data["actions"].shape == (32, 20)
    assert "observation.state" not in repacked
    assert data["image"]["right_wrist_0_rgb"].shape == (240, 240, 3)
    assert data["image"]["right_wrist_0_rgb"][0, 0, 0] == 63
    assert data["image"]["left_wrist_0_rgb"][0, 0, 0] == 191
    assert data["image_mask"] == dict(base_0_rgb=False, left_wrist_0_rgb=True, right_wrist_0_rgb=True)


def test_client_request_accepted_and_wrong_arm_count_rejected():
    example = policy.make_umi_bimanual_example()
    result = policy.UmiBimanualInputs(model.ModelType.PI05)(example)
    np.testing.assert_array_equal(result["current_tcp_pose9_gripper"], example["current_tcp_pose9_gripper"])
    example["current_tcp_pose9_gripper"] = np.zeros(10)
    with pytest.raises(ValueError, match=r"\[20\]"):
        policy.UmiBimanualInputs(model.ModelType.PI05)(example)
    with pytest.raises(KeyError):
        policy.UmiBimanualRepack()({"observation/image": np.zeros((240, 240, 3))})


def test_configurable_dataset_image_names():
    repack = policy.UmiBimanualRepack(right_image_key="right", left_image_key="left")
    result = repack({"right": 1, "left": 2})
    assert result == {"observation/image_right": 1, "observation/image_left": 2}


def test_training_and_serving_transform_chain():
    example = policy.make_umi_bimanual_example()
    reference = example["current_tcp_pose9_gripper"]
    reference[:3], reference[10:13] = [1, 2, 3], [-3, 1, 2]
    reference[[9, 19]] = [.5, 1.5]
    actions = np.tile(reference, (32, 1))
    actions[:, 0] += np.arange(32) * .01
    actions[:, 11] += np.arange(32) * .02
    inputs = policy.UmiBimanualInputs(model.ModelType.PI05)({**example, "actions": actions})
    relative = transforms.RelativeBimanualTcpPose9ColumnFromFirstAction()(inputs)
    stats = {"actions": normalize.NormStats(mean=np.zeros(20), std=np.ones(20), q01=np.zeros(20), q99=np.ones(20)*2)}
    normalized = transforms.UMIAction20Normalize(stats)(relative)["actions"]
    # Stand in for perfect model predictions, retaining the model's 32-D padding.
    model_output = {"actions": np.pad(normalized, ((0, 0), (0, 12))), "current_tcp_pose9_gripper": reference}
    output = transforms.UMIAction20Unnormalize(stats)(model_output)
    output = transforms.AbsoluteBimanualTcpPose9ColumnFromFirstAction()(output)
    output = policy.UmiBimanualOutputs()(output)
    assert output["actions"].shape == (32, 20)
    np.testing.assert_allclose(output["actions"], actions, atol=2e-6)

import types

import numpy as np
import pytest

from openpi import transforms
from openpi import transforms_umi as geom
from openpi.models import model as _model
from openpi.policies import umi_proprio_policy as policy


def _pose(x=0.0, angle=0.0, gripper=1.5):
    matrix = np.eye(4, dtype=np.float32)
    c, s = np.cos(angle), np.sin(angle)
    matrix[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    matrix[0, 3] = x
    return np.r_[geom.matrix_to_pose9(matrix), gripper].astype(np.float32)


def _sample():
    return {
        "observation/image": np.zeros((240, 240, 3), dtype=np.uint8),
        "observation/image_previous": np.full((240, 240, 3), 17, dtype=np.uint8),
        "prompt": "lift the bottle",
        "tcp_pose9_gripper_history": np.stack([_pose(0.4), _pose(0.5)]),
        "episode_start_tcp_pose9_gripper": _pose(0.1),
    }


def test_state_layout():
    state = policy.make_state(np.stack([_pose(0.4), _pose(0.5)]), _pose(0.1)).reshape(2, 16)
    np.testing.assert_allclose(state[:, :3], [[-0.1, 0, 0], [0, 0, 0]], atol=1e-6)
    np.testing.assert_allclose(state[:, 3:9], [[1, 0, 0, 0, 1, 0]] * 2)
    np.testing.assert_allclose(state[:, 9], 1.5)
    np.testing.assert_allclose(state[:, 10:], [[1, 0, 0, 0, 1, 0]] * 2)


def test_global_frame_invariance():
    history = np.stack([_pose(0.4, 0.2), _pose(0.5, 0.4)])
    start = _pose(0.1, -0.3)
    world = geom.pose9_to_matrix(_pose(2.0, 1.2)[:9])
    transformed_history = history.copy()
    transformed_history[:, :9] = geom.matrix_to_pose9(world @ geom.pose9_to_matrix(history[:, :9]))
    transformed_start = start.copy()
    transformed_start[:9] = geom.matrix_to_pose9(world @ geom.pose9_to_matrix(start[:9]))
    np.testing.assert_allclose(
        policy.make_state(history, start), policy.make_state(transformed_history, transformed_start), atol=1e-6
    )


def test_actions_reference_observation_and_roundtrip():
    sample = _sample()
    sample["actions"] = np.stack([_pose(0.6, 0.2, 1.0)] * 32)
    absolute = sample["actions"].copy()
    result = policy.UmiProprioInputs(_model.ModelType.PI05)(sample)
    np.testing.assert_allclose(result["actions"][:, 0], 0.1, atol=1e-6)
    np.testing.assert_allclose(result["actions"][:, 9], 1.0)
    recovered = geom.AbsoluteTcpPose9ColumnFromFirstAction()(result)
    np.testing.assert_allclose(recovered["actions"], absolute, atol=1e-6)


def test_training_and_inference_state_match():
    sample = _sample()
    training = dict(sample)
    training["observation.images.fisheye_img"] = training.pop("observation/image")
    training["action"] = np.stack([_pose(0.6)] * 32)
    training = policy.UmiProprioRepack()(training)
    transform = policy.UmiProprioInputs(_model.ModelType.PI05)
    train_result, infer_result = transform(training), transform(sample)
    np.testing.assert_array_equal(train_result["state"], infer_result["state"])
    for key in train_result["image"]:
        np.testing.assert_array_equal(train_result["image"][key], infer_result["image"][key])


def test_reference_mismatch_rejected():
    sample = {**_sample(), "current_tcp_pose9_gripper": _pose(2.0)}
    with pytest.raises(ValueError, match="latest history"):
        policy.UmiProprioInputs(_model.ModelType.PI05)(sample)


def test_bad_rotation_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        policy.make_state(np.zeros((2, 10)), _pose())


def test_history_stays_in_episode():
    class Dataset:
        hf_dataset = [
            {"episode_index": episode, "frame_index": frame, "observation.state": _pose(episode * 10 + frame)}
            for episode in range(2) for frame in range(5)
        ]

        def __len__(self):
            return len(self.hf_dataset)

        def __getitem__(self, index):
            return {"observation.images.fisheye_img": np.full((8, 8, 3), index, dtype=np.uint8)}

    dataset = policy.UmiHistoryDataset(Dataset(), 3)
    np.testing.assert_allclose(dataset[5]["tcp_pose9_gripper_history"][:, 0], [10, 10])
    np.testing.assert_allclose(dataset[6]["tcp_pose9_gripper_history"][:, 0], [10, 11])
    np.testing.assert_allclose(dataset[9]["tcp_pose9_gripper_history"][:, 0], [11, 14])
    assert dataset[9]["episode_start_tcp_pose9_gripper"][0] == 10


def test_normalization_preserves_action_rotations():
    stats = {
        name: transforms.NormStats(mean=np.zeros(n), std=np.ones(n), q01=-np.ones(n), q99=np.ones(n))
        for name, n in (("state", 32), ("actions", 10))
    }
    sample = _sample()
    sample["actions"] = np.stack([_pose(0.6, 0.3)] * 32)
    inputs = policy.UmiProprioInputs(_model.ModelType.PI05)(sample)
    rotations = inputs["actions"][:, 3:9].copy()
    result = policy.UmiProprioNormalize(stats)(inputs)
    assert result["state"].shape == (32,)
    np.testing.assert_array_equal(result["actions"][:, 3:9], rotations)
    inference = policy.UmiProprioInputs(_model.ModelType.PI05)(_sample())
    np.testing.assert_allclose(result["state"], policy.UmiProprioNormalize(stats)(inference)["state"])


def test_visual_only_stats_rejected():
    with pytest.raises(ValueError, match="Missing state"):
        policy.UmiProprioNormalize({"actions": object()})(_sample())


def test_state_is_passed_to_tokenizer():
    recorded = []
    tokenizer = types.SimpleNamespace(tokenize=lambda prompt, state: (recorded.append(state.copy()), None))
    inputs = policy.UmiProprioInputs(_model.ModelType.PI05)(_sample())
    transforms.TokenizePrompt(tokenizer, discrete_state_input=True)(inputs)
    np.testing.assert_array_equal(recorded[0], inputs["state"])


def test_temporal_image_slots_and_masks():
    result = policy.UmiProprioInputs(_model.ModelType.PI05)(_sample())
    np.testing.assert_array_equal(result["image"]["left_wrist_0_rgb"], 17)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], 0)
    assert result["image_mask"] == {
        "base_0_rgb": False, "left_wrist_0_rgb": True, "right_wrist_0_rgb": True,
    }


def test_missing_history_image_fails():
    sample = _sample()
    del sample["observation/image_previous"]
    with pytest.raises(KeyError, match="image_previous"):
        policy.UmiProprioInputs(_model.ModelType.PI05)(sample)


def test_image_and_state_share_indices_and_padding():
    class Dataset:
        hf_dataset = [
            {"episode_index": ep, "frame_index": frame, "observation.state": _pose(ep * 5 + frame)}
            for ep in range(2) for frame in range(5)
        ]

        def __init__(self):
            self.reads = []

        def __getitem__(self, index):
            self.reads.append(index)
            return {"observation.images.fisheye_img": np.full((8, 8, 3), index, dtype=np.uint8)}

    raw = Dataset()
    dataset = policy.UmiHistoryDataset(raw, 3)
    for index, old_index in ((5, 5), (6, 5), (9, 6)):
        raw.reads.clear()
        sample = dataset[index]
        assert raw.reads == ([index] if old_index == index else [index, old_index])
        np.testing.assert_array_equal(sample["observation/image_previous"], old_index)
        np.testing.assert_array_equal(sample["observation.images.fisheye_img"], index)
        np.testing.assert_array_equal(sample["tcp_pose9_gripper_history"][:, 0], [old_index, index])


def test_visual_only_adapter_still_has_one_image():
    result = policy.umi_policy.UmiInputs(_model.ModelType.PI05)(_sample())
    assert result["state"].size == 0
    assert result["image_mask"] == {
        "base_0_rgb": True, "left_wrist_0_rgb": False, "right_wrist_0_rgb": False,
    }

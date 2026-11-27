import numpy as np
import pytest

from openpi import transforms_umi_bimanual as transforms
from openpi.shared import normalize


def _actions():
    right_rot = [0, 1, 0, -1, 0, 0]
    left_rot = [0, -1, 0, 1, 0, 0]
    return np.array([
        [1, 2, 3, *right_rot, .5, -2, 4, 1, *left_rot, 1.5],
        [1, 2.1, 3, *left_rot, .7, -2, 3.8, 1, *right_rot, 1.7],
    ], dtype=np.float32)


def _stats(dim=20):
    return {"actions": normalize.NormStats(
        mean=np.zeros(dim), std=np.ones(dim), q01=np.zeros(dim),
        q99=np.arange(1, dim + 1, dtype=np.float32),
    )}


@pytest.mark.parametrize("batched", [False, True])
def test_independent_arm_references_and_roundtrip(batched):
    actions = _actions()
    if batched:
        actions = np.stack([actions, actions.copy()])
        actions[1, ..., :3] += 5
        actions[1, ..., 10:13] -= 3
    relative = transforms.RelativeBimanualTcpPose9ColumnFromFirstAction()({"actions": actions.copy()})["actions"]
    for start in (0, 10):
        np.testing.assert_allclose(relative[..., 0, start:start + 9],
                                   np.broadcast_to([0, 0, 0, 1, 0, 0, 0, 1, 0], relative[..., 0, start:start + 9].shape), atol=1e-6)
    np.testing.assert_allclose(relative[..., 1, 0], .1, atol=1e-6)
    np.testing.assert_allclose(relative[..., 1, 10], .2, atol=1e-6)
    np.testing.assert_array_equal(relative[..., [9, 19]], actions[..., [9, 19]])
    recovered = transforms.AbsoluteBimanualTcpPose9ColumnFromFirstAction()({
        "actions": relative, "current_tcp_pose9_gripper": actions[..., 0, :],
    })["actions"]
    np.testing.assert_allclose(recovered, actions, atol=1e-6)


def test_normalize_both_arms_preserves_rotations_and_model_padding():
    actions = np.pad(_actions(), ((0, 0), (0, 12)), constant_values=.42)
    normalized = transforms.UMIAction20Normalize(_stats())({"actions": actions.copy()})["actions"]
    for dim in (0, 1, 2, 9, 10, 11, 12, 19):
        np.testing.assert_allclose(normalized[:, dim], actions[:, dim] / (dim + 1 + 1e-6) * 2 - 1)
    untouched = [3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18, *range(20, 32)]
    np.testing.assert_array_equal(normalized[:, untouched], actions[:, untouched])
    recovered = transforms.UMIAction20Unnormalize(_stats())({"actions": normalized})["actions"]
    np.testing.assert_allclose(recovered, actions, atol=2e-6)


def test_reject_single_arm_stats_and_missing_or_wrong_reference():
    with pytest.raises(ValueError, match="20-D"):
        transforms.UMIAction20Normalize(_stats(10))({"actions": _actions()})
    absolute = transforms.AbsoluteBimanualTcpPose9ColumnFromFirstAction()
    with pytest.raises(ValueError, match="Missing"):
        absolute({"actions": _actions()})
    with pytest.raises(ValueError, match="reference"):
        absolute({"actions": _actions(), "current_tcp_pose9_gripper": np.zeros(10)})
    with pytest.raises(ValueError, match=">=20"):
        transforms.RelativeBimanualTcpPose9ColumnFromFirstAction()({"actions": np.zeros((32, 10))})


def test_serving_reference_is_independent_for_each_arm():
    relative = transforms.RelativeBimanualTcpPose9ColumnFromFirstAction()({"actions": _actions()})["actions"]
    reference = _actions()[0].copy()
    reference[:3] += [5, 6, 7]
    reference[10:13] += [-3, 2, -1]
    result = transforms.AbsoluteBimanualTcpPose9ColumnFromFirstAction()({
        "actions": np.pad(relative, ((0, 0), (0, 12))), "current_tcp_pose9_gripper": reference,
    })["actions"]
    expected = _actions()
    expected[:, :3] += [5, 6, 7]
    expected[:, 10:13] += [-3, 2, -1]
    assert result.shape == (2, 20)
    np.testing.assert_allclose(result, expected, atol=1e-6)

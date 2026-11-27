import numpy as np

from openpi import transforms_umi
from openpi.shared import normalize


def _stats() -> dict[str, normalize.NormStats]:
    return {
        "actions": normalize.NormStats(
            mean=np.zeros(10, dtype=np.float32),
            std=np.ones(10, dtype=np.float32),
            q01=np.zeros(10, dtype=np.float32),
            q99=np.ones(10, dtype=np.float32),
        )
    }


def test_relative_action_uses_first_action_and_preserves_gripper():
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    quarter_turn = np.array([0, 1, 0, -1, 0, 0], dtype=np.float32)
    actions = np.array(
        [
            [1, 2, 3, *identity, 0.1],
            [2, 4, 6, *quarter_turn, 0.2],
        ],
        dtype=np.float32,
    )

    result = transforms_umi.RelativeTcpPose9ColumnFromFirstAction()({"actions": actions})["actions"]

    np.testing.assert_allclose(result[0, :9], [0, 0, 0, 1, 0, 0, 0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(result[:, 9], actions[:, 9])


def test_relative_absolute_roundtrip_with_batch():
    identity = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    actions = np.array(
        [
            [[1, 2, 3, *identity, 0.1], [2, 4, 6, *identity, 0.2]],
            [[-1, 1, 0, *identity, 0.3], [-2, 1, 2, *identity, 0.4]],
        ],
        dtype=np.float32,
    )
    relative = transforms_umi.relative_pose9_from_first(actions[..., :9])
    absolute = transforms_umi.absolute_pose9_from_first(relative, actions[:, 0, :9])
    np.testing.assert_allclose(absolute, actions[..., :9], atol=1e-6)


def test_umi_normalization_roundtrip_keeps_rot6d():
    actions = np.array(
        [[[0.5, 0.25, 0.75, 1, 0, 0, 0, 1, 0, 0.5] * 1]],
        dtype=np.float32,
    ).reshape(1, 1, 10)
    normalized = transforms_umi.UMIAction10Normalize(_stats())({"actions": actions.copy()})["actions"]
    np.testing.assert_allclose(normalized[..., 3:9], actions[..., 3:9])
    recovered = transforms_umi.UMIAction10Unnormalize(_stats())({"actions": normalized})["actions"]
    np.testing.assert_allclose(recovered, actions, atol=2e-6)


def test_umi_unnormalization_handles_padded_model_output():
    actions = np.zeros((32, 32), dtype=np.float32)
    actions[:, :10] = 0.5
    result = transforms_umi.UMIAction10Unnormalize(_stats())({"actions": actions})["actions"]
    assert result.shape == (32, 32)
    np.testing.assert_allclose(result[:, 3:9], 0.5)
    np.testing.assert_allclose(result[:, 10:], 0.0)

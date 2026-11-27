"""Check RTC prefixes against the checkpoint's actual UMI output transforms."""

from types import SimpleNamespace
import numpy as np
import pytest
from numpy.testing import assert_allclose
from openpi import transforms_umi as single
from openpi import transforms_umi_bimanual as dual
from openpi.policies.rtc_umi import normalize_prefix


def pose(xyz, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.r_[xyz, [c, s, 0.0, -s, c, 0.0]]


@pytest.mark.parametrize("count", [1, 2])
def test_rebase_normalize_existing_output_roundtrip(count):
    reference = np.concatenate(
        [np.r_[pose([1 + i, 2 - i, 0.4], 0.7 + i), 1.0] for i in range(count)]
    )
    absolute = np.tile(reference, (32, 1))
    for arm in range(count):
        absolute[:, 10 * arm : 10 * arm + 9] = pose([0.2 + arm, 0.5, 0.8], -0.4 + arm)
        absolute[:, 10 * arm + 9] = 2.2 + arm * 0.1
    stats = SimpleNamespace(q01=np.full(10 * count, -2.0), q99=np.full(10 * count, 3.0))
    normalized = normalize_prefix(absolute, reference, stats, 32)
    assert normalized.shape == (32, 32)
    assert_allclose(normalized[:, 10 * count :], 0)
    data = dict(actions=normalized, current_tcp_pose9_gripper=reference)
    if count == 1:
        data = single.UMIAction10Unnormalize({"actions": stats})(data)
        data = single.AbsoluteTcpPose9ColumnFromFirstAction()(data)
    else:
        data = dual.UMIAction20Unnormalize({"actions": stats})(data)
        data = dual.AbsoluteBimanualTcpPose9ColumnFromFirstAction()(data)
    assert_allclose(data["actions"], absolute, atol=5e-7)
    # Guidance to the currently observed TCP is identity, independently per arm.
    same = normalize_prefix(reference[None], reference, stats, 32)
    for arm in range(count):
        assert_allclose(
            same[0, 10 * arm + 3 : 10 * arm + 9], [1, 0, 0, 0, 1, 0], atol=1e-6
        )


def test_wrong_checkpoint_statistics_rejected():
    reference = np.r_[pose([0, 0, 0], 0), 1.0]
    stats = SimpleNamespace(q01=np.zeros(20), q99=np.ones(20))
    with pytest.raises(ValueError, match="quantiles"):
        normalize_prefix(reference[None], reference, stats, 32)


def test_adapter_preserves_api_and_masks_only_real_action_dimensions(monkeypatch):
    import sys
    from types import ModuleType
    import jax
    from openpi.policies.rtc_umi import RTCUmiPolicy
    import openpi.models

    model_module = ModuleType("openpi.models.model")
    model_module.Observation = SimpleNamespace(from_dict=lambda value: value)
    monkeypatch.setitem(sys.modules, "openpi.models.model", model_module)
    monkeypatch.setattr(openpi.models, "model", model_module, raising=False)
    reference = np.r_[pose([0.2, 0.3, 0.4], 0.8), 1.0].astype(np.float32)
    absolute = np.tile(np.r_[pose([0.1, 0.2, 0.5], 0.2), 2.0], (32, 1)).astype(
        np.float32
    )
    stats = SimpleNamespace(q01=np.full(10, -2.0), q99=np.full(10, 3.0))
    captured = {}

    def input_transform(request):
        assert "rtc" not in request
        assert request["tcp_pose9_gripper_history"].shape == (2, 10)
        assert request["episode_start_tcp_pose9_gripper"].shape == (10,)
        return dict(
            state=np.zeros(32, np.float32),
            current_tcp_pose9_gripper=request["current_tcp_pose9_gripper"],
        )

    def output_transform(data):
        data = single.UMIAction10Unnormalize({"actions": stats})(data)
        return single.AbsoluteTcpPose9ColumnFromFirstAction()(data)

    base = SimpleNamespace(
        _input_transform=input_transform,
        _output_transform=output_transform,
        _rng=jax.random.key(0),
        infer=lambda request: {"legacy": True},
    )
    policy = object.__new__(RTCUmiPolicy)
    policy.base = base
    policy.stats = {"actions": stats}
    policy.num_robots = 1
    policy.dim = 32
    policy.horizon = 32
    policy.num_steps = 10
    policy.max_guidance = 5.0

    def sample(rng, observation, **kwargs):
        captured.update(kwargs)
        return kwargs["prefix"]

    policy.sample = sample
    request = dict(
        current_tcp_pose9_gripper=reference,
        tcp_pose9_gripper_history=np.stack([reference, reference]),
        episode_start_tcp_pose9_gripper=reference.copy(),
        rtc=dict(version=1, prefix_absolute=absolute, weights=np.ones(32)),
    )
    result = policy.infer(request)
    assert result["rtc_version"] == 1
    assert_allclose(result["actions"], absolute, atol=5e-7)
    assert_allclose(captured["weights"][0, :, :10], 1)
    assert_allclose(captured["weights"][0, :, 10:], 0)
    assert policy.infer({"current_tcp_pose9_gripper": reference}) == {"legacy": True}
    assert "rtc" in request  # Adapter does not consume caller's protocol data.

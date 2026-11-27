"""Offline tests: no cameras, robot bridge, or motor connection is opened."""

from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from umi.real_world.bimanual_policy import configure_gripper_conversion, decode_actions
from umi.real_world.rtc_policy import (
    ActionPlan,
    RTCExecutor,
    encode_absolute,
    make_request_plan,
    response_on_time,
    soft_weights,
    RTCProprioPolicyClient,
)


@pytest.mark.parametrize("count", [1, 2])
def test_gripper_calibration_and_pose_roundtrip(count):
    configs = [
        dict(
            width_to_rad=dict(
                min_abs_width_m=0.04 + i * 0.01,
                max_abs_width_m=0.15 + i * 0.02,
                min_rad=0.0,
                max_rad=3.5,
            )
        )
        for i in range(count)
    ]
    options = {}
    scales, inverse = configure_gripper_conversion(configs, 1000 / 30, 0.03, options)
    rng = np.random.default_rng(4)
    actions = rng.normal(size=(32, 7 * count)) * 0.2
    actions[:, 6::7] = 0.1
    wire = encode_absolute(actions, scales, options)
    restored = decode_actions(wire, np.zeros((count, 6)), inverse, False, options)
    assert_allclose(restored, actions, atol=1e-7)
    assert not np.isclose(wire[0, 9], actions[0, 6] * 1000 / 30)


def test_plan_slerp_and_endpoint_hold():
    action = np.zeros((2, 7))
    action[1, [0, 5, 6]] = [2.0, np.pi / 2, 0.12]
    plan = ActionPlan([1.0, 3.0], action)
    assert_allclose(
        plan.sample([0.0, 2.0, 4.0])[:, [0, 5, 6]],
        [[0, 0, 0], [1, np.pi / 4, 0.06], [2, np.pi / 2, 0.12]],
    )


def test_replace_preserves_already_committed_target():
    old = np.zeros((2, 7))
    old[1, 0] = 10
    plan = ActionPlan([0.0, 10.0], old)
    committed = plan.sample([3.0])[0]
    new = np.zeros((6, 7))
    new[:, 0] = 100
    plan.replace_future(np.arange(2.0, 8.0), new, 4.0, 3.0)
    assert_allclose(plan.sample([3.0])[0], committed)
    assert plan.sample([4.0])[0, 0] == 100
    with pytest.raises(ValueError, match="committed"):
        plan.replace_future(np.arange(2.0, 8.0), new, 3.0, 3.0)


def test_soft_weights_and_request_uses_actual_time():
    weights = soft_weights(8, 2, 6)
    assert_allclose(weights[:2], 1)
    assert np.all(np.diff(weights[1:6]) < 0)
    assert_allclose(weights[6:], 0)
    plan = ActionPlan([10.0, 12.0], np.zeros((2, 7)))
    times, frozen, rtc = make_request_plan(
        plan, 10.0, 10.1, 0.1, 32, 0.2, 0.15, 10.3, np.array([10.0]), {}
    )
    assert frozen == 5  # First mutable point is 10.5, beyond 10.45 deadline.
    assert_allclose(times, np.arange(32) * 0.1 + 10.0)
    assert_allclose(rtc["weights"][:frozen], 1)
    assert response_on_time(times, frozen, 10.3, 0.15, 10.4)
    assert not response_on_time(times, frozen, 10.4, 0.15, 10.4)
    assert not response_on_time(times, frozen, 10.1, 0.15, 10.5)
    with pytest.raises(TimeoutError):
        make_request_plan(
            plan, 10.0, 14.0, 0.1, 32, 0.2, 0.15, 10.3, np.array([10.0]), {}
        )


class Robot:
    def __init__(self):
        self.calls = []

    def schedule_waypoint(self, **kwargs):
        self.calls.append(kwargs)


class Gripper:
    def __init__(self):
        self.calls = []

    def move_gripper(self, width, **kwargs):
        self.calls.append((width, kwargs))


def fake_env():
    return SimpleNamespace(
        robots=[Robot(), Robot()],
        grippers=[Gripper(), Gripper()],
        robots_config=[dict(robot_action_latency=0.1), dict(robot_action_latency=0.2)],
        grippers_config=[
            dict(gripper_type="custom", gripper_action_latency=0.05),
            dict(gripper_type="custom", gripper_action_latency=0.1),
        ],
        action_accumulator=None,
    )


def test_executor_routes_each_arm_and_gripper_with_latency():
    env = fake_env()
    executor = RTCExecutor(env)
    action = np.arange(14, dtype=float)
    executor.submit(action, 2.0, 1.0)
    assert_allclose(env.robots[0].calls[0]["pose"], action[:6])
    assert_allclose(env.robots[1].calls[0]["pose"], action[7:13])
    assert env.robots[0].calls[0]["target_time"] == 1.9
    assert env.robots[1].calls[0]["target_time"] == 1.8
    executor.dispatch(1.91)
    assert not env.grippers[0].calls
    assert env.grippers[1].calls[0][0] == 13
    executor.dispatch(1.96)
    assert env.grippers[0].calls[0][0] == 6
    with pytest.raises(ValueError):
        executor.submit(action, 2.0, 1.0)
    with pytest.raises(TimeoutError):
        executor.submit(action, 2.1, 2.0)


def test_executor_debug_never_sends_commands():
    env = fake_env()
    executor = RTCExecutor(env, debug=True)
    executor.submit(np.zeros(14), 2.0, 1.0)
    executor.dispatch(3.0)
    assert not any(r.calls for r in env.robots)
    assert not any(g.calls for g in env.grippers)


def test_gripper_stall_drops_obsolete_motor_updates():
    env = fake_env()
    executor = RTCExecutor(env)
    executor.submit(np.zeros(14), 2.0, 1.0)
    executor.submit(np.ones(14), 2.1, 1.0)
    executor.dispatch(3.0)
    assert len(env.grippers[0].calls) == 1
    assert env.grippers[0].calls[0][0] == 1


def test_rtc_timing_supports_explicit_slow_execution():
    from policy_client_pi_rtc import validate_rtc_timing

    assert validate_rtc_timing(
        dict(fps=15, training_fps=30, execution_fps=15, time_scale=2), 15
    ) == (30.0, 15.0, 2.0)
    assert validate_rtc_timing(dict(fps=30), 30) == (30.0, 30.0, 1.0)
    with pytest.raises(ValueError, match="timing metadata mismatch"):
        validate_rtc_timing(
            dict(fps=15, training_fps=30, execution_fps=15, time_scale=1), 15
        )
    with pytest.raises(ValueError, match="control_fps"):
        validate_rtc_timing(
            dict(fps=15, training_fps=30, execution_fps=15, time_scale=2), 30
        )


def test_rtc_gripper_offset_closes_once_and_respects_each_limit():
    from policy_client_pi_rtc import apply_gripper_close_offset

    actions = np.zeros((2, 14), dtype=float)
    actions[:, 6] = [0.080, 0.046]
    actions[:, 13] = [0.070, 0.050]
    original = actions.copy()

    tuned = apply_gripper_close_offset(
        actions, 0.005, [(0.043, 0.150), (0.040, 0.155)]
    )

    assert_allclose(tuned[:, 6], [0.075, 0.043])
    assert_allclose(tuned[:, 13], [0.065, 0.045])
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("offset", [-0.001, np.nan, np.inf])
def test_rtc_gripper_offset_rejects_invalid_values(offset):
    from policy_client_pi_rtc import apply_gripper_close_offset

    with pytest.raises(ValueError, match="finite and non-negative"):
        apply_gripper_close_offset(np.zeros((1, 7)), offset, [None])


class StartupClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class StartupThread:
    def __init__(self, alive=True):
        self.alive = alive

    def is_alive(self):
        return self.alive


def test_initial_observation_waits_for_custom_gripper_history(monkeypatch):
    from umi.real_world import client_deployment as entry

    clock = StartupClock()
    monkeypatch.setattr(entry, "time", clock)

    class Gripper:
        thread = StartupThread()
        calls = 0

        def get_all_state(self):
            self.calls += 1
            count = 0 if self.calls < 3 else 2
            return {
                "gripper_timestamp": np.arange(count, dtype=float),
                "gripper_position": np.arange(count, dtype=float),
            }

    expected = {"timestamp": np.array([1.0])}
    env = SimpleNamespace(
        grippers=[Gripper()],
        grippers_config=[{"gripper_type": "custom"}],
        get_obs=lambda: expected,
    )
    assert entry.wait_for_initial_observation(env, timeout=1.0) is expected
    assert env.grippers[0].calls == 3


def test_initial_observation_reports_dead_custom_gripper_thread():
    from policy_client_pi_rtc import wait_for_initial_observation

    gripper = SimpleNamespace(
        thread=StartupThread(alive=False),
        get_all_state=lambda: {},
    )
    env = SimpleNamespace(
        grippers=[gripper],
        grippers_config=[{"gripper_type": "custom"}],
    )
    with pytest.raises(RuntimeError, match="gripper 0 control thread stopped"):
        wait_for_initial_observation(env)


def test_initial_observation_timeout_has_hardware_context(monkeypatch):
    from umi.real_world import client_deployment as entry

    clock = StartupClock()
    monkeypatch.setattr(entry, "time", clock)
    gripper = SimpleNamespace(
        thread=StartupThread(),
        get_all_state=lambda: {
            "gripper_timestamp": np.array([]),
            "gripper_position": np.array([]),
        },
    )
    env = SimpleNamespace(
        grippers=[gripper],
        grippers_config=[{"gripper_type": "custom"}],
    )
    with pytest.raises(TimeoutError, match="check camera, robot, and gripper"):
        entry.wait_for_initial_observation(env, timeout=0.11, poll_interval=0.05)


def test_proprio_rtc_client_uses_explicit_history_payload():
    client = RTCProprioPolicyClient.__new__(RTCProprioPolicyClient)
    client.num_robots, client.prompt = 1, "pick"
    client.options = {
        "image_keys": ["observation/image"],
        "state_key": "current_tcp_pose9_gripper",
    }
    client.reset_episode()
    state = np.r_[0, 0, 0, 1, 0, 0, 0, 1, 0, 0.5].astype(np.float32)
    old = np.full((8, 8, 3), 10, np.uint8)
    current = np.full((8, 8, 3), 20, np.uint8)
    client.set_history(old, np.stack([state, state]), 1.0, 1.1)
    payload = client.build_payload([current], state)
    assert client.proprioception is True
    assert "tcp_pose9_gripper_history" in payload
    assert "episode_start_tcp_pose9_gripper" in payload
    np.testing.assert_array_equal(payload["observation/image_previous"], current)


def test_episode_commits_while_inference_pending_and_discards_on_stop(
    monkeypatch, tmp_path
):
    import policy_client_pi_rtc as entry
    import umi.real_world.rtc_policy as scheduling

    class Clock:
        now = 100.0

        def time(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    monkeypatch.setattr(entry, "time", clock)
    monkeypatch.setattr(scheduling, "time", clock)
    env = fake_env()
    env.get_obs = lambda: dict(timestamp=np.array([clock.now - 0.01]))
    env.start_episode = lambda started: None
    env.ended = False
    submitted_during_inference = []

    class Client:
        metadata = {"action_horizon": 32}

        def call_rtc(self, images, state, rtc):
            raw = np.tile(state, (32, 1))
            raw[:, 0] = 0.1
            raw[:, 10] = 0.2
            return raw

    class Future:
        def __init__(self, fn, args):
            self.fn, self.args, self.ready = fn, args, clock.now + 0.09

        def done(self):
            return clock.now >= self.ready

        def result(self):
            clock.now = max(clock.now, self.ready)
            return self.fn(*self.args)

    class Pool:
        current = None

        def submit(self, fn, *args):
            self.current = Future(fn, args)
            return self.current

    pool = Pool()
    for robot in env.robots:
        original = robot.schedule_waypoint

        def schedule(original=original, **kwargs):
            submitted_during_inference.append(
                pool.current is not None and not pool.current.done()
            )
            original(**kwargs)

        robot.schedule_waypoint = schedule
    state = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0.1], 2).astype(np.float32)
    monkeypatch.setattr(
        entry,
        "build_observation",
        lambda *a: (
            [np.zeros((2, 2, 3), np.uint8)] * 2,
            state.copy(),
            np.zeros((2, 6)),
            np.ones(2) * 0.1,
        ),
    )
    helpers = SimpleNamespace(
        read_episode_keys=lambda keys: {"s"} if clock.now >= 100.8 else set(),
        solve_table_collision=lambda *a: None,
        solve_sphere_collision=lambda *a: None,
        finish_episode_recording=lambda env: setattr(env, "ended", True),
    )
    for robot_config in env.robots_config:
        robot_config["height_threshold"] = 0
    args = SimpleNamespace(
        frequency=20.0,
        delivery_margin=0.05,
        execute=True,
        image_resolution=2,
        initial_delay=0.11,
        delay_margin=0.02,
        max_duration=10.0,
        max_timesteps=100,
        steps_per_inference=3,
        max_hold=0.5,
        gripper_close_offset_m=0.005,
        output=tmp_path,
    )
    quit_requested = entry.run_episode(
        env, Client(), pool, args, {}, np.ones(2), np.ones(2), None, helpers
    )
    assert env.ended and not quit_requested
    assert any(submitted_during_inference)
    targets = env.robots[0].calls
    assert any(r["pose"][0] > 0.01 for r in targets)
    assert max(r["target_time"] for r in targets) <= 100.8 + 0.3
    widths = [width for width, _ in env.grippers[0].calls]
    assert min(widths) == pytest.approx(0.095)
    assert max(widths) == pytest.approx(0.1)

"""Opt-in UMI deployment with real-time chunking."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
import time
import numpy as np
import yaml
from umi.real_world.bimanual_policy import (
    build_observation,
    decode_actions,
    configure_gripper_conversion,
    policy_options,
    validate_hardware_config,
)
from umi.real_world.rtc_policy import (
    ActionPlan,
    RTCPolicyClient,
    RTCProprioPolicyClient,
    RTCExecutor,
    make_request_plan,
    response_on_time,
)
from umi.real_world.proprio_policy_client import paired_history
from umi.real_world.client_deployment import wait_for_initial_observation, warmup_requests


def validate_rtc_timing(metadata, control_fps):
    try:
        legacy_fps = float(metadata["fps"])
        training_fps = float(metadata.get("training_fps", legacy_fps))
        execution_fps = float(metadata.get("execution_fps", legacy_fps))
        time_scale = float(
            metadata.get("time_scale", training_fps / execution_fps)
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise ValueError("RTC server has invalid timing metadata") from exc
    if (
        not np.isfinite([training_fps, execution_fps, time_scale]).all()
        or min(training_fps, execution_fps, time_scale) <= 0
        or not np.isclose(execution_fps, control_fps)
        or not np.isclose(time_scale, training_fps / execution_fps)
    ):
        raise ValueError(
            "RTC timing metadata mismatch: control_fps must equal server execution_fps"
        )
    return training_fps, execution_fps, time_scale


def apply_gripper_close_offset(actions, offset_m, width_limits):
    """Reduce decoded gripper widths without crossing hardware lower limits."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] % 7 != 0:
        raise ValueError(
            f"Expected [T, 7 * num_robots] decoded actions, got {actions.shape}"
        )
    if not np.isfinite(offset_m) or offset_m < 0:
        raise ValueError("Gripper close offset must be finite and non-negative")
    count = actions.shape[1] // 7
    if len(width_limits) != count:
        raise ValueError("Need one gripper width limit entry per robot")

    tuned = actions.copy()
    for index, limits in enumerate(width_limits):
        minimum = 0.0 if limits is None else float(limits[0])
        width_index = index * 7 + 6
        tuned[:, width_index] = np.maximum(
            tuned[:, width_index] - offset_m, minimum
        )
    return tuned


def build_request_observation(client, obs, count, resolution, scales, options):
    images, state, poses, widths = build_observation(
        obs, count, resolution, scales, options
    )
    if getattr(client, "proprioception", False):
        previous_images, history = paired_history(
            obs, resolution, scales, options, count=count
        )
        timestamps = np.asarray(obs["timestamp"], dtype=float)
        if timestamps.shape != (2,):
            raise ValueError("Proprio RTC requires exactly two observation timestamps")
        client.set_history(
            previous_images, history, float(timestamps[0]), float(timestamps[-1])
        )
    return images, state, poses, widths


def run_episode(
    env,
    client,
    pool,
    args,
    options,
    scales,
    inverse,
    key_counter,
    helpers,
):
    dt = 1 / args.frequency
    horizon = int(client.metadata["action_horizon"])
    lead = (
        max(
            [float(c.get("robot_action_latency", 0)) for c in env.robots_config]
            + [float(c.get("gripper_action_latency", 0)) for c in env.grippers_config]
        )
        + args.delivery_margin
    )
    executor = RTCExecutor(env, debug=not args.execute)
    if getattr(client, "proprioception", False):
        client.reset_episode()
    obs = env.get_obs()
    images, state, poses, widths = build_request_observation(
        client, obs, len(scales), args.image_resolution, scales, options
    )
    initial = np.concatenate([np.r_[p, w] for p, w in zip(poses, widths)])
    started = time.time()
    origin = started + lead + dt
    plan = ActionPlan([float(obs["timestamp"][-1])], initial[None])
    committed = -np.inf
    future = None
    request = None
    last_request = -np.inf
    accepted_end = None
    gripper_offset_start = None
    delays = deque([args.initial_delay], maxlen=10)
    next_tick = origin
    step = 0
    try:
        env.start_episode(started)
        while True:
            now = time.time()
            executor.dispatch(now)
            keys = helpers.read_episode_keys(key_counter)
            if (
                "q" in keys
                or "s" in keys
                or now - started >= args.max_duration
                or step >= args.max_timesteps
            ):
                return "q" in keys
            if future is not None and future.done():
                raw = future.result()
                completed = time.time()
                times, frozen, request_poses, requested_at = request
                elapsed = completed - requested_at
                delays.append(elapsed + args.delay_margin)
                on_time = response_on_time(times, frozen, completed, lead, committed)
                if on_time:
                    action = decode_actions(raw, request_poses, inverse, False, options)
                    tuned_widths = apply_gripper_close_offset(
                        action,
                        args.gripper_close_offset_m,
                        options.get("gripper_width_limits", [None] * len(scales)),
                    )
                    for target, tuned_target in zip(action, tuned_widths):
                        for i, rc in enumerate(env.robots_config):
                            helpers.solve_table_collision(
                                target[7 * i : 7 * i + 6],
                                tuned_target[7 * i + 6],
                                rc.get("height_threshold", 0.01),
                            )
                        helpers.solve_sphere_collision(
                            target.reshape(len(scales), 7)[:, :6], env.robots_config
                        )
                    plan.replace_future(times, action, times[frozen], committed)
                    accepted_end = float(times[-1])
                    if gripper_offset_start is None:
                        gripper_offset_start = float(times[frozen])
                else:
                    print(
                        "RTC: late response discarded; previous plan continues. Delay estimate updated."
                    )
                future = None
            now = time.time()
            if accepted_end is not None and now + lead > accepted_end + args.max_hold:
                raise TimeoutError(
                    "RTC plan exhausted; ending episode after bounded final-target hold"
                )
            # At most one newly committed waypoint per cycle; never burst catch-up.
            if now >= next_tick - lead:
                if now > next_tick - lead + dt:
                    raise TimeoutError(
                        "RTC control loop missed a full timestep; stopping submission"
                    )
                planned_action = plan.sample([next_tick])[0]
                if (
                    gripper_offset_start is not None
                    and next_tick >= gripper_offset_start
                ):
                    action = apply_gripper_close_offset(
                        planned_action[None],
                        args.gripper_close_offset_m,
                        options.get("gripper_width_limits", [None] * len(scales)),
                    )[0]
                else:
                    action = planned_action
                executor.submit(action, next_tick, now)
                committed = next_tick
                next_tick += dt
                step += 1
            now = time.time()
            if future is None and now - last_request >= args.steps_per_inference * dt:
                obs = env.get_obs()
                images, state, poses, widths = build_request_observation(
                    client, obs, len(scales), args.image_resolution, scales, options
                )
                observed = float(obs["timestamp"][-1])
                times, frozen, rtc = make_request_plan(
                    plan,
                    observed,
                    time.time(),
                    dt,
                    horizon,
                    max(delays),
                    lead,
                    committed,
                    scales,
                    options,
                )
                requested_at = time.time()
                request = (times, frozen, poses.copy(), requested_at)
                future = pool.submit(client.call_rtc, images, state, rtc)
                last_request = requested_at
            time.sleep(0.002)
    finally:
        # Drain only issued short-horizon commands. A pending result never crosses episodes.
        try:
            executor.drain()
            if future is not None:
                # Finish/discard the response before R or a new C can reuse the socket.
                future.result()
        finally:
            helpers.finish_episode_recording(env)


def apply_rtc_config(args, config):
    """Resolve CLI > YAML > built-in defaults; execution stays CLI-only."""
    rtc = config.get("rtc", {})
    if not isinstance(rtc, dict):
        raise ValueError("rtc must be a YAML mapping")
    defaults = {
        "steps_per_inference": 8,
        "initial_delay": 0.2,
        "delay_margin": 0.05,
        "delivery_margin": 0.05,
        "max_hold": 0.5,
        "max_duration": 30.0,
        "max_timesteps": 600,
    }
    unknown = set(rtc) - set(defaults)
    if unknown:
        raise ValueError(f"Unknown rtc settings: {sorted(unknown)}")
    for name, default in defaults.items():
        value = getattr(args, name)
        if value is None:
            value = rtc.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"rtc.{name} must be numeric")
        if isinstance(default, int) and not isinstance(value, int):
            raise ValueError(f"rtc.{name} must be an integer")
        setattr(args, name, value)
    policy = config.get("policy", {})
    if args.frequency is None:
        args.frequency = float(policy.get("control_fps", 30.0))
    if args.image_resolution is None:
        value = policy.get("image_resolution", 240)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("policy.image_resolution must be an integer")
        args.image_resolution = value


def main(*, proprioception=False):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot-config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--frequency",
        type=float,
        help="Defaults to policy.control_fps; must match RTC server",
    )
    p.add_argument(
        "--steps-per-inference",
        type=int,
        default=None,
        help="Minimum control steps between requests; defaults to rtc.steps_per_inference (8)",
    )
    p.add_argument(
        "--initial-delay",
        type=float,
        default=None,
        help="Estimated request latency; defaults to rtc.initial_delay (0.2 seconds)",
    )
    p.add_argument("--delay-margin", type=float, default=None, help="Overrides YAML rtc.delay_margin; default 0.05")
    p.add_argument("--delivery-margin", type=float, default=None, help="Overrides YAML rtc.delivery_margin; default 0.05")
    p.add_argument("--max-hold", type=float, default=None, help="Overrides YAML rtc.max_hold; default 0.5")
    p.add_argument("--max-duration", type=float, default=None, help="Overrides YAML rtc.max_duration; default 30.0")
    p.add_argument("--max-timesteps", type=int, default=None, help="Overrides YAML rtc.max_timesteps; default 600")
    p.add_argument("--image-resolution", type=int, default=None, help="Overrides YAML policy.image_resolution; default 240")
    p.add_argument(
        "--gripper-close-offset-mm",
        type=float,
        default=0.0,
        help="Subtract this many millimetres from model gripper targets; default 0",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Submit actions; default is prediction/virtual execution only",
    )
    p.add_argument("--init-joints", action="store_true")
    args = p.parse_args()
    with open(args.robot_config) as f:
        config = yaml.safe_load(f)
    try:
        apply_rtc_config(args, config)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))
    policy = config.get("policy", {})
    robots = config["robots"]
    grippers = config["grippers"]
    from umi.real_world.mvs_camera_config import camera_configs_from_yaml

    cameras = camera_configs_from_yaml(config)
    count = len(robots)
    validate_hardware_config(robots, grippers)
    if any(r["robot_type"] != "franka" for r in robots):
        p.error("RTC client currently supports Franka only")
    options = policy_options(policy, count)
    scales, inverse = configure_gripper_conversion(
        grippers,
        policy.get("hardware_to_model_gripper_scale", 1000 / 30),
        policy.get("model_to_hardware_gripper_scale", 0.03),
        options,
    )
    if (
        not math.isfinite(args.gripper_close_offset_mm)
        or args.gripper_close_offset_mm < 0
    ):
        p.error("--gripper-close-offset-mm must be finite and non-negative")
    args.gripper_close_offset_m = args.gripper_close_offset_mm / 1000.0
    values = [
        args.frequency,
        args.initial_delay,
        args.delay_margin,
        args.delivery_margin,
        args.max_hold,
        args.max_duration,
    ]
    if (
        any(not math.isfinite(v) or v <= 0 for v in values)
        or args.steps_per_inference < 1
        or args.max_timesteps < 1
        or args.image_resolution < 1
    ):
        p.error("RTC timing, sizes and counts must be positive and finite")
    if cameras is None or len(cameras) != count:
        p.error("Configure one wrist camera per arm")
    if proprioception and count != 1:
        p.error("The current UMI proprioception training contract is single-arm only")
    if options["state_key"] != "current_tcp_pose9_gripper":
        p.error("RTC UMI uses the existing current_tcp_pose9_gripper state key")
    for rc in robots:
        if rc.get("joints_init") is not None:
            joints = np.asarray(rc["joints_init"], dtype=float)
            if joints.shape != (7,) or not np.isfinite(joints).all():
                p.error("joints_init must contain seven finite joint angles in radians")
        duration = float(rc.get("joints_init_duration", 5.0))
        if not math.isfinite(duration) or duration <= 0:
            p.error("joints_init_duration must be positive and finite")
    for entry, key in [(r, "robot_action_latency") for r in robots] + [
        (g, "gripper_action_latency") for g in grippers
    ]:
        value = float(entry.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            p.error(f"{key} must be nonnegative and finite")
    if policy.get("action_is_relative", False):
        p.error("RTC server returns absolute actions; set action_is_relative: false")
    args.output = args.output.expanduser()
    args.output.mkdir(parents=True, exist_ok=False)
    # Hardware imports remain local. Existing environment owns camera/robot workers.
    from multiprocessing.managers import SharedMemoryManager
    from umi.real_world.bimanual_umi_env import BimanualUmiEnv
    from umi.real_world.keystroke_counter import KeystrokeCounter
    import policy_client_pi_episodes as helpers

    client_class = RTCProprioPolicyClient if proprioception else RTCPolicyClient
    client = client_class(
        policy["server_url"],
        policy.get("prompt"),
        timeout=float(policy.get("request_timeout_s", 60)),
        num_robots=count,
        policy_config=policy,
    )
    try:
        meta = client.metadata
        if (
            meta.get("rtc_version") != 1
            or meta.get("num_robots") != count
            or meta.get("is_relative") is not False
            or meta.get("action_dim") != 10 * count
            or meta.get("image_keys") != options["image_keys"]
            or bool(meta.get("proprioception", False)) is not proprioception
        ):
            raise ValueError("RTC server/model contract mismatch")
        training_fps, execution_fps, time_scale = validate_rtc_timing(
            meta, args.frequency
        )
        if not args.steps_per_inference < int(meta["action_horizon"]):
            raise ValueError(
                "steps-per-inference must be shorter than prediction horizon"
            )
        print(
            "RTC mode:",
            (
                "LIVE"
                if args.execute
                else "DEBUG: no policy/home/opening commands; hardware controllers still start and gripper zero resets"
            ),
        )
        print(
            f"RTC timing: training={training_fps:g} Hz, "
            f"execution={execution_fps:g} Hz, time_scale={time_scale:g}x"
        )
        print(f"Gripper close offset: {args.gripper_close_offset_mm:g} mm")
        history_gap = float(policy.get("proprio_history_seconds", 0.1))
        if proprioception and (
            not math.isfinite(history_gap) or not 0 < history_gap <= 1.0
        ):
            raise ValueError("policy.proprio_history_seconds must be within (0, 1]")
        if proprioception and (
            not isinstance(meta.get("history_steps"), int)
            or meta["history_steps"] < 1
            or not np.isclose(history_gap * training_fps, meta["history_steps"])
        ):
            raise ValueError(
                "policy.proprio_history_seconds does not match checkpoint history_steps/fps"
            )
        history_steps = history_gap * args.frequency if proprioception else 1
        with SharedMemoryManager() as shm, KeystrokeCounter() as keys, BimanualUmiEnv(
            output_dir=str(args.output),
            video_only=True,
            robots_config=robots,
            grippers_config=grippers,
            cameras_config=cameras,
            frequency=args.frequency,
            obs_image_resolution=(args.image_resolution, args.image_resolution),
            obs_float32=False,
            init_joints=args.execute
            and (
                args.init_joints
                or any(r.get("joints_init") is not None for r in robots)
            ),
            shm_manager=shm,
            camera_obs_latency=0.17,
            camera_down_sample_steps=(
                [history_steps, 0] if proprioception else 1
            ),
            robot_down_sample_steps=history_steps,
            gripper_down_sample_steps=history_steps,
            camera_obs_horizon=2,
            robot_obs_horizon=2,
            gripper_obs_horizon=2,
            enable_multi_cam_vis=True,
        ) as env, ThreadPoolExecutor(max_workers=1) as pool:
            initial = []
            durations = []
            for robot, rc in zip(env.robots, robots):
                helpers.wait_robot_ready(robot)
                initial.append(
                    np.asarray(
                        rc.get("joints_init")
                        if rc.get("joints_init") is not None
                        else robot.get_state()["ActualQ"]
                    )
                )
                durations.append(float(rc.get("joints_init_duration", 5)))
            if args.execute:
                helpers.restore_initial_gripper_opening(env, grippers)
            def warmup_rtc(obs):
                images, state, _, _ = build_request_observation(
                    client, obs, count, args.image_resolution, scales, options
                )
                horizon = int(meta["action_horizon"])
                client.call_rtc(
                    images,
                    state,
                    dict(
                        version=1,
                        prefix_absolute=np.tile(state, (horizon, 1)),
                        weights=np.ones(horizon, np.float32),
                    ),
                )

            try:
                if not warmup_requests(env, warmup_rtc, lambda: helpers.read_episode_keys(keys)):
                    return
            finally:
                if getattr(client, "proprioception", False):
                    client.reset_episode()
            print(
                "RTC ready. C=start, S=end, R=home while idle, Q=quit. Terminal keys may require Enter."
            )
            while True:
                pressed = helpers.read_episode_keys(keys)
                if "q" in pressed:
                    break
                if "r" in pressed and args.execute:
                    helpers.return_to_initial_joints(env, shm, initial, durations)
                    helpers.restore_initial_gripper_opening(env, grippers)
                if "c" in pressed:
                    if run_episode(
                        env,
                        client,
                        pool,
                        args,
                        options,
                        scales,
                        inverse,
                        keys,
                        helpers,
                    ):
                        break
                    print("Episode finished. C=start, R=home, Q=quit.")
                time.sleep(0.01)
    finally:
        client.close()


if __name__ == "__main__":
    main()

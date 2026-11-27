"""Warmup and receding-horizon timing shared by the OpenPI deployment clients."""

import time

import numpy as np


def wait_for_initial_observation(env, timeout=10.0, poll_interval=0.05):
    """Wait for the camera, pose, and gripper interpolation buffers."""
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        try:
            ready = True
            for i, (gripper, config) in enumerate(zip(env.grippers, env.grippers_config)):
                if config.get("gripper_type", "wsg") != "custom":
                    continue
                thread = getattr(gripper, "thread", None)
                if thread is not None and not thread.is_alive():
                    raise RuntimeError(f"Custom gripper {i} control thread stopped during startup")
                state = gripper.get_all_state()
                if min(np.size(state.get("gripper_timestamp", ())),
                       np.size(state.get("gripper_position", ()))) < 2:
                    ready = False
            if ready:
                return env.get_obs()
        except (IndexError, ValueError) as exc:
            last_error = exc
        time.sleep(poll_interval)
    raise TimeoutError(f"No initial robot/camera/gripper observation within {timeout:g}s; check camera, robot, and gripper") from last_error


def warmup_requests(env, infer, read_keys, count=2):
    """Compile/warm the server with real observations and discard every response."""
    obs = wait_for_initial_observation(env)
    for i in range(count):
        if "q" in read_keys():
            return False
        if i:
            obs = env.get_obs()
        print(f"[Warmup] Inference {i + 1}/{count}; response will be discarded.", flush=True)
        infer(obs)
    # Discard C/R typed while the request was running. Q still quits.
    return "q" not in read_keys()


def warmup_pi(env, client, resolution, scales, options, read_keys):
    from umi.real_world.bimanual_policy import build_observation

    proprio = hasattr(client, "set_history")

    def infer(obs):
        images, state, _, _ = build_observation(obs, len(scales), resolution, scales, options)
        if proprio:
            from umi.real_world.proprio_policy_client import paired_history

            previous, history = paired_history(obs, resolution, scales, options, count=len(scales))
            timestamps = obs["timestamp"]
            client.set_history(previous, history, float(timestamps[0]), float(timestamps[-1]))
        client.call_server(images, state)

    try:
        return warmup_requests(env, infer, read_keys)
    finally:
        if proprio:
            client.reset_episode()


def action_delivery_margin(robots, grippers, delivery_s=0.05):
    delays = (
        [float(cfg.get("robot_action_latency", 0.0)) for cfg in robots]
        + [float(cfg.get("gripper_action_latency", 0.0)) for cfg in grippers]
    )
    if any(not np.isfinite(d) or d < 0 for d in delays):
        raise ValueError("Action latencies must be finite and non-negative")
    return max(delays, default=0.0) + delivery_s


def select_live_waypoints(predicted_steps, observation_time, dt, now, delivery_margin,
                          committed_until, limit, episode_deadline):
    """Select the earliest future, uncommitted rows using the observation clock."""
    times = observation_time + np.arange(predicted_steps, dtype=np.float64) * dt
    cutoff = max(now + delivery_margin, committed_until + 1e-6)
    indices = np.flatnonzero((times > cutoff) & (times <= episode_deadline))[:limit]
    return indices, times[indices]


def blend_action_rows(target_times, chunks, dt):
    """Average predictions for the same wall-clock target, within half a tick."""
    from umi.real_world.bimanual_policy import average_actions

    rows = []
    for target_time in target_times:
        candidates = []
        for start, actions in chunks:
            step = round((target_time - start) / dt)
            if 0 <= step < len(actions):
                candidates.append(actions[step])
        weights = np.exp(-0.01 * np.arange(len(candidates), dtype=float))
        rows.append(average_actions(np.asarray(candidates), weights / weights.sum()))
    return np.concatenate(rows, axis=0)


def wait_pending_actions(env, last_action_end_time, timeout=20.0):
    """Let submitted actions finish; never clear a worker-owned queue."""
    deadline = time.monotonic() + max(timeout, last_action_end_time - time.time() + 5.0)
    custom_grippers = [
        grip for grip, config in zip(env.grippers, env.grippers_config)
        if config.get('gripper_type') == 'custom'
    ]
    while True:
        if any(not robot.is_alive() for robot in env.robots):
            raise RuntimeError('Robot worker stopped')
        if any(not grip.thread.is_alive() for grip in custom_grippers):
            raise RuntimeError('Custom gripper worker stopped')
        if (time.time() >= last_action_end_time
                and all(robot.input_queue.empty() for robot in env.robots)
                and not any(grip.has_pending_waypoints() for grip in custom_grippers)):
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Previous actions did not drain')
        time.sleep(0.02)

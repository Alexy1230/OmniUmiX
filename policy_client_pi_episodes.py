"""
Legacy episode scheduling for one or two arms.

Inference runs every loop. steps-per-inference gates submission by loop count,
while exec-horizon-steps independently caps each submitted action chunk.
Warmup responses are discarded; episodes record first-person videos only.
The shared environment schedules each accepted gripper waypoint with its arm.

"""
import os
import sys
import select
import time
import atexit
import yaml
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager

os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.client_deployment import warmup_pi, wait_pending_actions
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.keystroke_counter import KeystrokeCounter
from umi.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.real_world.bimanual_policy import (
    PolicyClient, policy_options, validate_hardware_config,
    configure_gripper_conversion, build_observation, decode_actions, average_actions,
)
def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.635, 0])  # Left base expressed in right base; TODO:

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']

            if distance < threshold:
                if distance < 1e-8:
                    raise ValueError('Gripper sphere centers coincide; cannot choose an avoidance direction')
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

def read_episode_keys(key_counter):
    """Accept either case from pynput, the OpenCV window, or terminal + Enter."""
    keys = {str(getattr(event, 'char', '') or '').lower()
            for event in key_counter.get_press_events()}
    code = cv2.pollKey()
    if code >= 0:
        keys.add(chr(code & 0xff).lower())
    # Canonical terminal input becomes readable after Enter (also works over SSH).
    if sys.stdin is not None and sys.stdin.isatty():
        if select.select([sys.stdin], [], [], 0)[0]:
            keys.update(sys.stdin.readline().strip().lower())
    return keys & {'c', 's', 'r', 'q'}


def finish_episode_recording(env):
    env.end_episode()


def episode_stop_reason(key_counter, started_at, max_duration, step, max_timesteps):
    """Check before inference and again before submitting a returned prediction."""
    events = read_episode_keys(key_counter)
    if 'q' in events:
        return 'quit'
    if 's' in events:
        return 'human stop'
    if time.time() - started_at >= max_duration:
        return 'max duration'
    if step >= max_timesteps:
        return 'max timesteps'
    return None


def wait_robot_ready(robot):
    # The package start_wait() only checks that the process is alive; joint
    # initialization may still be running and no observations may exist yet.
    timeout = float(robot.joints_init_duration or 0.0) + 20.0
    if not robot.ready_event.wait(timeout) or not robot.is_alive():
        raise RuntimeError('Robot did not become ready after joint initialization')


def return_to_initial_joints(env, shm_manager, initial_joints, duration):
    """Reuse the existing worker's joint-init path without competing arm commands."""
    # Home sequentially; the other robot keeps its existing hold controller.
    for i, (joints, seconds) in enumerate(zip(initial_joints, duration)):
        old = env.robots[i]
        print(f'robot{i}: returning to initial joints over {seconds:g}s...', flush=True)
        old.stop(wait=True)
        robot = FrankaInterpolationController(
            shm_manager=shm_manager, robot_ip=old.robot_ip,
            robot_port=old.robot_port, frequency=old.frequency,
            joints_init=joints, joints_init_duration=seconds,
            launch_timeout=seconds + 20.0,
            receive_latency=old.receive_latency, tx_flange_tcp=old.tx_flange_tcp,
            soft_real_time=old.soft_real_time, verbose=old.verbose,
            get_max_k=old.ring_buffer.get_max_k)
        robot.Kx, robot.Kxd = old.Kx.copy(), old.Kxd.copy()
        robot.start(wait=False)
        env.robots[i] = robot
        wait_robot_ready(robot)
    time.sleep(0.2)
    print('Joint return complete. Waiting for C.', flush=True)


def restore_initial_gripper_opening(env, grippers_config):
    """Restore each configured absolute opening, keeping both motor zeros unchanged."""
    for i, gc in enumerate(grippers_config):
        if 'initial_position' in gc:
            width = float(gc['initial_position'])
            if gc.get('gripper_type', 'wsg') == 'custom':
                env.grippers[i].move_gripper(width_m=width, force_limit_nm=float(gc.get('force_limit_nm', 1.5)))
            else:
                env.grippers[i].schedule_waypoint(pos=width, target_time=time.time() + 0.5)
            print(f'gripper{i}: initial opening requested {width*1000:.1f} mm (zero unchanged).', flush=True)


# ============ 主函数 ============

@click.command()
@click.option('--server-url', '-s', default=None, help='OpenPI WebSocket URL; defaults to policy.server_url in robot config')
@click.option('--prompt', default=None, help='Task prompt; defaults to policy.prompt in robot config')
@click.option('--request-timeout', default=None, type=float, help='WebSocket timeout; defaults to policy.request_timeout_s')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot-config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--camera-reorder', '-cr', default=None, help='Optional comma-separated camera indices')
@click.option('--vis-camera-idx', default=0, type=int, help='Which camera to visualize')
@click.option('--init-joints', '-j', is_flag=True, default=False, help='Initialize joints once at startup; automatic when YAML joints_init is present')
@click.option('--steps-per-inference', '-si', default=1, type=int, help='Submit a new chunk every N loop iterations; inference still runs each iteration')
@click.option('--max-duration', '-md', default=2000000, type=float, help='Max duration per episode (seconds)')
@click.option('--max-timesteps', '-mt', default=500, type=int, help='Max loop iterations per episode, including iterations without submission')
@click.option('--frequency', '-f', default=None, type=float, help='Control frequency; defaults to policy.control_fps')
@click.option('--obs-image-resolution', default=None, type=int, help='Image size; defaults to policy.image_resolution')
@click.option('--exec-horizon-steps', default=None, type=int, help='Actions scheduled per inference; defaults to policy.exec_horizon_steps')
@click.option('--no-mirror', '-nm', is_flag=True, default=False, help='Disable mirror')
@click.option('--mirror-swap', is_flag=True, default=False, help='Enable mirror swap')
@click.option('--temporal-agg', is_flag=True, default=False, help='Enable temporal aggregation')
@click.option('--ensemble-steps', type=int, default=8, help='Ensemble steps for temporal agg')
def main(server_url, prompt, request_timeout, output, robot_config,
         camera_reorder, vis_camera_idx, init_joints,
         steps_per_inference, max_duration, max_timesteps,
         frequency, obs_image_resolution, exec_horizon_steps, no_mirror, mirror_swap,
         temporal_agg, ensemble_steps):


    # Load robot config
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    policy_config = robot_config_data.get('policy', {})
    server_url = server_url or policy_config.get('server_url')
    prompt = prompt if prompt is not None else policy_config.get('prompt', '')
    request_timeout = request_timeout or float(policy_config.get('request_timeout_s', 30.0))
    frequency = frequency or float(policy_config.get('control_fps', 30.0))
    obs_image_resolution = obs_image_resolution or int(policy_config.get('image_resolution', 240))
    exec_horizon_steps = exec_horizon_steps or int(policy_config.get('exec_horizon_steps', 32))
    action_is_relative = bool(policy_config.get('action_is_relative', True))
    hardware_to_model_gripper_scale = policy_config.get('hardware_to_model_gripper_scale', 1000.0 / 30.0)
    model_to_hardware_gripper_scale = policy_config.get('model_to_hardware_gripper_scale', 30.0 / 1000.0)
    obs_res = (obs_image_resolution, obs_image_resolution)

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    from umi.real_world.mvs_camera_config import camera_configs_from_yaml
    cameras_config = camera_configs_from_yaml(robot_config_data)
    num_robots = len(robots_config)
    validate_hardware_config(robots_config, grippers_config)
    options = policy_options(policy_config, num_robots)
    hardware_to_model_gripper_scale, model_to_hardware_gripper_scale = configure_gripper_conversion(
        grippers_config, hardware_to_model_gripper_scale, model_to_hardware_gripper_scale, options)

    if (not np.isfinite(frequency) or frequency <= 0 or steps_per_inference <= 0
            or exec_horizon_steps <= 0 or max_timesteps <= 0
            or not np.isfinite(max_duration) or max_duration <= 0 or ensemble_steps <= 0):
        raise click.UsageError('Use positive frequency/steps/duration')

    dt = 1 / frequency
    down_sample_steps = 1
    for i, (rc, gc) in enumerate(zip(robots_config, grippers_config)):
        if rc.get('joints_init') is not None:
            joints = np.asarray(rc['joints_init'], dtype=float)
            duration = float(rc.get('joints_init_duration', 5.0))
            if joints.shape != (7,) or not np.isfinite(joints).all() or not np.isfinite(duration) or duration <= 0:
                raise click.UsageError(f'robot{i}: joints_init needs seven finite radians and positive duration')
        if 'initial_position' in gc:
            width = float(gc['initial_position'])
            if not np.isfinite(width) or not 0 <= width <= 0.20:
                raise click.UsageError(f'gripper{i}: initial_position must be metres within [0, 0.20]')

    if camera_reorder is None:
        camera_indices = None if cameras_config is not None else list(range(num_robots))
    else:
        camera_indices = [int(x) for x in camera_reorder.split(',')]
    if cameras_config is not None:
        selected = list(range(len(cameras_config))) if camera_indices is None else camera_indices
        if len(selected) < num_robots or len(set(selected)) != len(selected) or any(i < 0 or i >= len(cameras_config) for i in selected):
            raise click.UsageError('Provide one distinct wrist camera per robot, in robot order')
    print(f'Model contract: {options}')

    print("=" * 60)
    print("UMI Policy Client (Naive Episodes)")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Number of robots: {num_robots}")
    for i, rc in enumerate(robots_config):
        print(f"Robot {i} endpoint: {rc['robot_ip']}:{int(rc.get('robot_port', 4242))}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Submission interval: {steps_per_inference} loop iterations; inference runs every loop")
    print(f"Execution horizon: up to {exec_horizon_steps} accepted action rows per submission")
    print("=" * 60)

    print('[Startup] Connecting to policy server...', flush=True)
    policy_client = PolicyClient(server_url, prompt=prompt, timeout=request_timeout,
                                 num_robots=num_robots, policy_config=policy_config)
    atexit.register(policy_client.close)

    print('[Startup] Starting keyboard listener and hardware environment...', flush=True)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                video_only=True,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=False,
                camera_reorder=camera_indices,
                cameras_config=cameras_config,
                init_joints=(init_joints or any(rc.get('joints_init') is not None for rc in robots_config)),
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,
                # downsample
                camera_down_sample_steps=down_sample_steps,
                robot_down_sample_steps=down_sample_steps,
                gripper_down_sample_steps=down_sample_steps,
                # obs horizon (默认值，可以根据模型调整)
                camera_obs_horizon=2,
                robot_obs_horizon=2,
                gripper_obs_horizon=2,
                no_mirror=no_mirror,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:

            cv2.setNumThreads(2)
            print("Waiting for camera...")
            time.sleep(1.0)

            initial_joints, home_duration = [], []
            for i, (robot, rc) in enumerate(zip(env.robots, robots_config)):
                wait_robot_ready(robot)
                initial_joints.append(np.asarray(rc['joints_init'] if rc.get('joints_init') is not None
                                                else robot.get_state()['ActualQ'], dtype=float).copy())
                home_duration.append(float(rc.get('joints_init_duration', 5.0)))
                print(f'robot{i} initial joints for R: {initial_joints[-1]}')
            restore_initial_gripper_opening(env, grippers_config)

            if not warmup_pi(
                env, policy_client, obs_image_resolution, hardware_to_model_gripper_scale, options,
                lambda: read_episode_keys(key_counter),
            ):
                return

            print('C starts; S ends; R returns to initial joints and restores gripper opening while idle; Q quits.')
            print('After an episode, submitted actions finish and the arm holds. Homing requires R.')

            print("\n" + "=" * 60)
            print("Ready! Press 'C' to start policy, 'Q' to quit")
            print("=" * 60)

            action_dim = 7 * num_robots
            all_time_actions = None
            episode_idx = 0

            while True:
                print("\nWaiting for start command...")
                print('C: start here; R: return to initial joints; Q: quit. Terminal input requires Enter.', flush=True)

                while True:
                    obs = env.get_obs()
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1].copy()

                    # show states
                    text = 'C: start; R: home; Q: quit'
                    cv2.putText(vis_img, text, (10, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, color=(255, 255, 255))
                    cv2.imshow('default', vis_img[..., ::-1])

                    press_events = read_episode_keys(key_counter)
                    if 'q' in press_events:
                        print("Exiting...")
                        return
                    if 'r' in press_events:
                        return_to_initial_joints(env, shm_manager, initial_joints, home_duration)
                        restore_initial_gripper_opening(env, grippers_config)
                        # Homing never starts inference; require a fresh C afterwards.
                        if 'q' in read_episode_keys(key_counter):
                            return
                        continue
                    if 'c' in press_events:
                        print('Starting episode...', flush=True)
                        break
                    time.sleep(0.05)

                # ========== policy control loop ==========
                quit_requested = False
                last_action_end_time = 0.0
                try:
                    episode_idx += 1
                    all_time_actions = None
                    print(f'Episode {episode_idx}')
                    print("\n" + "=" * 60)
                    print("Policy control started!")
                    print("Press 'S' to stop")
                    print("=" * 60)

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    inference_idx = steps_per_inference

                    while True:
                        reason = episode_stop_reason(
                            key_counter, eval_t_start, max_duration, iter_idx, max_timesteps)
                        if reason:
                            print(f'Episode stopped: {reason}')
                            quit_requested = reason == 'quit'
                            break
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        images, current_state, current_poses6, current_widths = build_observation(
                            obs, num_robots, obs_image_resolution,
                            hardware_to_model_gripper_scale, options)


                        # Server-client communication
                        s = time.time()
                        raw_action = policy_client.call_server(images, current_state)
                        response_received_at = time.time()


                        reason = episode_stop_reason(
                            key_counter, eval_t_start, max_duration, iter_idx, max_timesteps)
                        if reason:
                            print(f'Episode stopped after inference: {reason}')
                            quit_requested = reason == 'quit'
                            break
                        if raw_action is None or np.array(raw_action).size == 0:
                            print("Warning: Received empty action from server!")
                            iter_idx += 1
                            inference_idx += 1
                            precise_wait(t_cycle_end - frame_latency)
                            continue

                        inference_time = response_received_at - s
                        print(f'Network latency: {inference_time:.3f}s')

                        if raw_action is None:
                            print("No action from server, skipping...")
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += 1
                            continue

                        action = decode_actions(raw_action, current_poses6,
                                                model_to_hardware_gripper_scale,
                                                action_is_relative, options)
                        action_horizon = action.shape[0]

                        if temporal_agg:
                            required_steps = max_timesteps + action_horizon
                            if all_time_actions is None or iter_idx == 0:
                                all_time_actions = np.full(
                                    (max_timesteps, required_steps, action_dim), np.nan
                                )
                            elif all_time_actions.shape[1] < required_steps:
                                all_time_actions = np.pad(all_time_actions, (
                                    (0, 0),
                                    (0, required_steps - all_time_actions.shape[1]),
                                    (0, 0)), constant_values=np.nan)
                            all_time_actions[
                                iter_idx, iter_idx:iter_idx + action_horizon
                            ] = action

                        if inference_idx >= steps_per_inference:
                            inference_idx = 0

                            # temporal aggregation
                            if temporal_agg:
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                                target_pose_list = []
                                for i in range(action_horizon):
                                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
                                    actions_populated = np.all(np.isfinite(actions_for_curr_step), axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                                    if len(actions_for_curr_step) > 0:
                                        k = -0.01
                                        exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                        exp_weights = exp_weights / exp_weights.sum()
                                        weighted_action = average_actions(actions_for_curr_step, exp_weights)
                                        target_pose_list.append(weighted_action)
                                    else:
                                        target_pose_list.append(action[[i]])

                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = action

                            # collision check
                            for target_pose in this_target_poses:
                                for robot_idx in range(num_robots):
                                    solve_table_collision(
                                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                        gripper_width=target_pose[robot_idx * 7 + 6],
                                        height_threshold=robots_config[robot_idx].get('height_threshold', 0.01)
                                    )

                                if num_robots > 1:
                                    solve_sphere_collision(
                                        ee_poses=target_pose.reshape([num_robots, -1]),
                                        robots_config=robots_config
                                    )

                            # latency compensation
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            # env.exec_actions subtracts robot_action_latency.
                            # Keep 50 ms for local delivery after that subtraction.
                            # curr_time below already includes elapsed inference time.
                            action_exec_latency = max(
                                float(rc.get('robot_action_latency', 0.0))
                                for rc in robots_config) + 0.05
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            prediction_steps = len(action_timestamps)
                            kept_indices = np.flatnonzero(is_new)
                            filtered_steps = prediction_steps - len(kept_indices)
                            used_fallback = len(kept_indices) == 0


                            if np.sum(is_new) == 0:
                                selected_indices = np.array([prediction_steps - 1])
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil(
                                    (curr_time + action_exec_latency - eval_t_start) / dt
                                ))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(f'Over budget: {action_timestamp - curr_time:.3f}s')
                                action_timestamps = np.array([action_timestamp])
                            else:
                                selected_indices = kept_indices
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # receding horizons
                            exec_start = 0
                            exec_end = exec_horizon_steps
                            this_target_poses = this_target_poses[exec_start:exec_end]
                            action_timestamps = action_timestamps[exec_start:exec_end]
                            selected_indices = selected_indices[exec_start:exec_end]
                            remaining_steps = 1 if used_fallback else len(kept_indices)
                            print(
                                f"[Timing] step={iter_idx} predicted={prediction_steps} "
                                f"filtered={filtered_steps} eligible={len(kept_indices)} "
                                f"horizon_dropped={remaining_steps - len(selected_indices)} "
                                f"selected_indices={selected_indices.tolist()} (0-based) "
                                f"fallback={used_fallback}; "
                                f"obs_age={curr_time - obs_timestamps[-1]:.3f}s "
                                f"selection_margin={action_exec_latency:.3f}s dt={dt:.4f}s"
                            )
                            if len(selected_indices):
                                for i, rc in enumerate(robots_config):
                                    lead = action_timestamps[0] - float(rc.get('robot_action_latency', 0)) - time.time()
                                    target = this_target_poses[0, 7*i:7*i+6]
                                    delta = np.linalg.norm(target[:3] - current_poses6[i, :3]) * 1000
                                    print(f'[Timing] robot{i}: target distance={delta:.1f}mm compensated_lead={lead*1000:.1f}ms')

                            last_action_end_time = max(
                                last_action_end_time, float(np.max(action_timestamps)))
                            accepted = env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            print(f"Step {iter_idx}: selected={len(this_target_poses)} "
                                      f"submitted={len(accepted)} actions")

                        # visualization
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1].copy()
                        text = f'Step: {iter_idx}, Time: {time.monotonic() - t_start:.1f}s'
                        cv2.putText(vis_img, text, (10, 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))
                        cv2.imshow('default', vis_img[..., ::-1])

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("\nInterrupted!")
                    quit_requested = True
                finally:
                    try:
                        print('[Episode] Waiting for submitted arm and gripper actions to finish...', flush=True)
                        wait_pending_actions(env, last_action_end_time)
                    finally:
                        print('[Episode] Finalizing recording...', flush=True)
                        finish_episode_recording(env)

                if quit_requested:
                    return
                # Discard start/home keys pressed during the ending trial or cleanup.
                if 'q' in read_episode_keys(key_counter):
                    return
                print('Episode ended. Holding position. R: home; C: start here; Q: quit.', flush=True)


if __name__ == '__main__':
    main()

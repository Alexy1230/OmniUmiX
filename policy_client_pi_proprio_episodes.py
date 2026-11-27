"""
OpenPI PC client for one or two Frankas.
[Only support frankas and ur]

"""
import os
import sys
import select
import time
import atexit
from collections import deque
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

from umi.real_world.proprio_policy_client import ProprioPolicyClient, paired_history
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.client_deployment import (
    wait_pending_actions,
    action_delivery_margin, blend_action_rows, select_live_waypoints, warmup_pi,
)
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.keystroke_counter import KeystrokeCounter
from umi.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from umi.real_world.bimanual_policy import (
    policy_options, validate_hardware_config,
    configure_gripper_conversion, build_observation, decode_actions,
)


def apply_gripper_close_offset(actions, offset_m, width_limits):
    """Reduce decoded gripper widths without crossing hardware lower limits."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] % 7 != 0:
        raise ValueError(
            f'Expected [T, 7 * num_robots] decoded actions, got {actions.shape}'
        )
    if not np.isfinite(offset_m) or offset_m < 0:
        raise ValueError('Gripper close offset must be finite and non-negative')

    num_robots = actions.shape[1] // 7
    if len(width_limits) != num_robots:
        raise ValueError('Need one gripper width limit entry per robot')

    tuned = actions.copy()
    for robot_idx, limits in enumerate(width_limits):
        min_width = 0.0 if limits is None else float(limits[0])
        width_idx = robot_idx * 7 + 6
        tuned[:, width_idx] = np.maximum(
            tuned[:, width_idx] - offset_m, min_width
        )
    return tuned


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
@click.option('--steps-per-inference', '-si', default=None, type=int, help='Minimum request interval in control steps and maximum submitted rows; defaults to policy.steps_per_inference or 1')
@click.option('--max-duration', '-md', default=2000000, type=float, help='Max duration per episode (seconds)')
@click.option('--max-timesteps', '-mt', default=500, type=int, help='Max steps per episode')
@click.option('--frequency', '-f', default=None, type=float, help='Control frequency; defaults to policy.control_fps')
@click.option('--obs-image-resolution', default=None, type=int, help='Image size; defaults to policy.image_resolution')
@click.option('--no-mirror', '-nm', is_flag=True, default=False, help='Disable mirror')
@click.option('--mirror-swap', is_flag=True, default=False, help='Enable mirror swap')
@click.option('--temporal-agg', is_flag=True, default=False, help='Enable temporal aggregation')
@click.option('--ensemble-steps', type=int, default=8, help='Ensemble steps for temporal agg')
@click.option('--gripper-close-offset-mm', type=float, default=0.0, show_default=True,
              help='Subtract this many millimetres from every decoded gripper target')
def main(server_url, prompt, request_timeout, output, robot_config,
         camera_reorder, vis_camera_idx, init_joints,
         steps_per_inference, max_duration, max_timesteps,
         frequency, obs_image_resolution, no_mirror, mirror_swap,
         temporal_agg, ensemble_steps, gripper_close_offset_mm,
         ):


    # Load robot config
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    policy_config = dict(robot_config_data.get('policy', {}))
    history_gap = float(policy_config.get('proprio_history_seconds', 0.1))
    if policy_config.get('action_is_relative', False) is not False:
        raise click.UsageError('Proprio server returns absolute actions: set action_is_relative: false')
    policy_config['action_is_relative'] = False
    if not np.isfinite(history_gap) or not 0 < history_gap <= 1.0:
        raise click.UsageError('proprio_history_seconds must be within (0, 1] and match training')
    server_url = server_url or policy_config.get('server_url')
    prompt = prompt if prompt is not None else policy_config.get('prompt', '')
    request_timeout = request_timeout or float(policy_config.get('request_timeout_s', 30.0))
    frequency = float(policy_config.get('control_fps', 30.0)) if frequency is None else frequency
    obs_image_resolution = obs_image_resolution or int(policy_config.get('image_resolution', 240))
    if steps_per_inference is None:
        steps_per_inference = int(policy_config.get('steps_per_inference', 1))
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
            or max_timesteps <= 0
            or not np.isfinite(max_duration) or max_duration <= 0 or ensemble_steps <= 0):
        raise click.UsageError('Use positive frequency/steps/duration')
    if not np.isfinite(gripper_close_offset_mm) or gripper_close_offset_mm < 0:
        raise click.UsageError(
            '--gripper-close-offset-mm must be finite and non-negative'
        )
    gripper_close_offset_m = gripper_close_offset_mm / 1000.0

    dt = 1 / frequency
    delivery_margin = action_delivery_margin(robots_config, grippers_config)
    down_sample_steps = history_gap * frequency
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
    print("UMI Policy Client)")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Number of robots: {num_robots}")
    for i, rc in enumerate(robots_config):
        print(f"Robot {i} endpoint: {rc['robot_ip']}:{int(rc.get('robot_port', 4242))}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Steps per inference: {steps_per_inference}")
    print(f"Gripper close offset: {gripper_close_offset_mm:g} mm")
    print("=" * 60)

    print('[Startup] Connecting to policy server...', flush=True)
    policy_client = ProprioPolicyClient(server_url, prompt=prompt, timeout=request_timeout,
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
                camera_down_sample_steps=[down_sample_steps, 0],
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
                    print(f'Episode {episode_idx}')
                    print("\n" + "=" * 60)
                    print("Policy control started!")
                    print("Press 'S' to stop")
                    print("=" * 60)

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    policy_client.reset_episode()

                    frame_latency = 1 / 60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")

                    iter_idx = 0
                    next_request_at = time.monotonic()
                    chunks = deque(maxlen=ensemble_steps)
                    episode_deadline = eval_t_start + max_duration

                    while True:
                        reason = episode_stop_reason(
                            key_counter, eval_t_start, max_duration, iter_idx, max_timesteps)
                        if reason:
                            print(f'Episode stopped: {reason}')
                            quit_requested = reason == 'quit'
                            break
                        remaining = next_request_at - time.monotonic()
                        if remaining > 0:
                            time.sleep(min(0.01, remaining))
                            continue
                        next_request_at = time.monotonic() + steps_per_inference * dt

                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        images, current_state, current_poses6, current_widths = build_observation(
                            obs, num_robots, obs_image_resolution,
                            hardware_to_model_gripper_scale, options)

                        previous_images, state_history = paired_history(
                            obs, obs_image_resolution, hardware_to_model_gripper_scale, options,
                            count=num_robots)
                        policy_client.set_history(previous_images, state_history, float(obs_timestamps[0]),
                                                  float(obs_timestamps[-1]))


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
                        if raw_action is None or np.asarray(raw_action).size == 0:
                            print("Warning: Received empty action from server!")
                            continue
                        print(f'Network latency: {response_received_at - s:.3f}s')

                        decoded_action = decode_actions(
                            raw_action, current_poses6,
                            model_to_hardware_gripper_scale,
                            action_is_relative, options)
                        action = apply_gripper_close_offset(
                            decoded_action,
                            gripper_close_offset_m,
                            options['gripper_width_limits'])
                        observed_at = float(obs_timestamps[-1])
                        indices, action_timestamps = select_live_waypoints(
                            len(action), observed_at, dt, time.time(), delivery_margin,
                            last_action_end_time, min(steps_per_inference, max_timesteps - iter_idx),
                            episode_deadline,
                        )
                        if temporal_agg:
                            chunks.append((observed_at, action))
                        if not len(indices):
                            print('[Timing] No future, uncommitted actions. Holding the last target.', flush=True)
                            continue
                        targets = (blend_action_rows(action_timestamps, chunks, dt)
                                   if temporal_agg else action[indices].copy())

                        for target in targets:
                            for arm, rc in enumerate(robots_config):
                                solve_table_collision(
                                    ee_pose=target[7 * arm:7 * arm + 6],
                                    gripper_width=target[7 * arm + 6],
                                    height_threshold=rc.get('height_threshold', 0.01),
                                )
                            if num_robots > 1:
                                solve_sphere_collision(target.reshape(num_robots, 7), robots_config)

                        # Keep the deadline if enqueue fails after a partial submission.
                        previous_end = last_action_end_time
                        last_action_end_time = float(action_timestamps[-1])
                        accepted = env.exec_actions(targets, action_timestamps, compensate_latency=True)
                        last_action_end_time = float(accepted[-1]) if len(accepted) else previous_end
                        iter_idx += len(accepted)
                        print(f'[Timing] predicted={len(action)} selected={indices.tolist()} '
                              f'submitted={len(accepted)} total={iter_idx}', flush=True)

                        # visualization
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1].copy()
                        text = f'Step: {iter_idx}, Time: {time.monotonic() - t_start:.1f}s'
                        cv2.putText(vis_img, text, (10, 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))
                        cv2.imshow('default', vis_img[..., ::-1])


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

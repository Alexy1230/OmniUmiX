"""
UMI PI05 episodes: C starts; S ends; R returns to initial joints; Q exits.

参考 eval_real.py 实现，使用 BimanualUmiEnv 环境

Usage:
    python umi_policy_client.py \
        --server-url http://<服务器IP>:8099 \
        --robot-config example/eval_robots_config.yaml \
        --output data_local/eval_output

架构：
    本地机器                                     服务器
    ┌─────────────────────────┐                ┌──────────────────┐
    │ BimanualUmiEnv          │                │                  │
    │   └─ 机器人/夹爪/相机   │                │  策略推理服务     │
    │                         │                │  (端口 8099)     │
    │ umi_policy_client.py    │ ──HTTP请求──▶  │                  │
    │   └─ 获取obs → 发送     │ ◀──返回动作──  │                  │
    │   └─ 执行动作           │                │                  │
    └─────────────────────────┘                └──────────────────┘


    cd /share/project/dcw/projects/openpi

export OPENPI_DATA_HOME=/share/project/dcw/projects/openpi/.openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.6

.venv/bin/python scripts/serve_policy.py \
  --port=8099 \
  policy:checkpoint \
  --policy.config=pi05_umi_lift_bottle \ (pi05_umi_adjust_slip_tube)
  --policy.dir=/share/project/dcw/projects/openpi/checkpoints/pi05_umi_lift_bottle/lift_bottle_lora/19999
  (/share/project/dcw/projects/openpi/checkpoints/pi05_umi_adjust_slip_tube/adjust_slip_tube_lora_20260904/39999)

  .venv/bin/python scripts/serve_policy.py   --port=8099   policy:checkpoint   --policy.config=lxy_pi05_umi_pick_and_place   --policy.dir=/share/project/lxy/openpi/checkpoints/lxy_pi05_umi_pick_and_place/pick_and_place_0929/19999

"""
import os
import sys
import select
import time
import atexit
import json
import yaml
import click
import cv2
import numpy as np
import scipy.spatial.transform as st
from websockets.sync.client import connect
from multiprocessing.managers import SharedMemoryManager

# 限制线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.keystroke_counter import KeystrokeCounter
from umi.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from reactive_diffusion_policy.common.space_utils import (
    ortho6d_to_rotation_matrix,
    pose_6d_to_pose_9d,
)
from reactive_diffusion_policy.common.action_utils import (
    relative_actions_to_absolute_actions,
)
from umi.common import openpi_msgpack_numpy
from scipy.spatial.transform import Rotation as R

# # 尝试导入 json_numpy
# json_numpy = None
# USE_JSON_NUMPY = True
# ============ 坐标转换函数 ============

def axis_angle_to_rpy(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle to RPY (euler angles xyz order)."""
    original_shape = axis_angle.shape[:-1]
    aa_flat = axis_angle.reshape(-1, 3)
    rpys = []
    for i in range(len(aa_flat)):
        rot = st.Rotation.from_rotvec(aa_flat[i])
        rpys.append(rot.as_euler('xyz'))
    return np.array(rpys).reshape(*original_shape, 3)


def rpy_to_axis_angle(rpy: np.ndarray) -> np.ndarray:
    """Convert RPY (euler angles xyz order) to axis-angle."""
    original_shape = rpy.shape[:-1]
    rpy_flat = rpy.reshape(-1, 3)
    axis_angles = []
    for i in range(len(rpy_flat)):
        rot = st.Rotation.from_euler('xyz', rpy_flat[i])
        axis_angles.append(rot.as_rotvec())
    return np.array(axis_angles).reshape(*original_shape, 3)


# ============ 碰撞检测函数 (来自 eval_real.py) ============

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    """检测并解决与桌面的碰撞"""
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
    """检测并解决两个机械臂之间的碰撞"""
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0])  # TODO: 根据实际情况调整

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
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal

                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))


# ============ 策略客户端类 ============

class PolicyClient:
    """OpenPI WebSocket transport with NumPy-aware MessagePack."""

    def __init__(self, server_url: str, prompt: str, timeout: float = 60.0):
        if server_url.startswith(('http://', 'https://')):
            raise ValueError("PI05 client requires ws:// or wss://, not the HTTP /dp URL")
        if not server_url.startswith(('ws://', 'wss://')):
            server_url = f'ws://{server_url}'
        self.server_url = server_url.rstrip('/')
        self.prompt = prompt
        self.timeout = timeout
        self.packer = openpi_msgpack_numpy.Packer()
        self.ws = connect(
            self.server_url,
            open_timeout=timeout,
            close_timeout=min(timeout, 5.0),
            compression=None,
            max_size=None,
        )
        metadata_message = self.ws.recv(timeout=timeout)
        self.metadata = openpi_msgpack_numpy.unpackb(metadata_message)
        keys = list(self.metadata) if isinstance(self.metadata, dict) else type(self.metadata).__name__
        print(f"OpenPI connected; metadata keys: {keys}")

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def call_server(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        payload = {
            'observation/image': np.asarray(image, dtype=np.uint8),
            'current_tcp_pose9_gripper': np.asarray(state, dtype=np.float32),
            'prompt': self.prompt,
        }
        try:
            self.ws.send(self.packer.pack(payload))
            message = self.ws.recv(timeout=self.timeout)
            if isinstance(message, str):
                raise RuntimeError(f"OpenPI server error: {message}")
            result = openpi_msgpack_numpy.unpackb(message)

            # ==============================================================================================
            raw_action = result.get('actions', result.get('action')) if isinstance(result, dict) else result
            # ==============================================================================================

            action = np.asarray(raw_action, dtype=np.float32)
            if action.ndim == 1:
                action = action[None]
            if action.ndim != 2 or action.shape[1] != 10:
                raise ValueError(f"Expected PI05 actions shaped [T, 10], got {action.shape}")
            if not np.all(np.isfinite(action)):
                raise ValueError("PI05 returned non-finite actions")
            return action
        except Exception as e:
            print(f"[PolicyClient] Server error: {e}")
            return None


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
    """Avoid the environment's timestamps[-1] on a trial with no actions."""
    if env.obs_accumulator is None:
        return
    actions = env.action_accumulator
    obs_times = env.obs_accumulator.timestamps
    if (actions is None or len(actions.timestamps) == 0
            or not obs_times or any(len(t) == 0 for t in obs_times.values())):
        env.camera.stop_recording()
        env.obs_accumulator = None
        env.action_accumulator = None
        print('[Episode] No complete observation/action samples; replay entry skipped.', flush=True)
        return
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


def wait_pending_actions(env, last_action_end_time, timeout=20.0):
    """Let submitted actions finish; never clear a worker-owned queue."""
    deadline = time.monotonic() + max(timeout, last_action_end_time - time.time() + 5.0)
    while True:
        if any(not robot.is_alive() for robot in env.robots):
            raise RuntimeError('Robot worker stopped')
        if all(robot.input_queue.empty() for robot in env.robots) and time.time() >= last_action_end_time:
            return
        if time.monotonic() > deadline:
            raise TimeoutError('Previous robot actions did not drain; not starting another trial')
        time.sleep(0.02)


def wait_robot_ready(robot):
    # The package start_wait() only checks that the process is alive; joint
    # initialization may still be running and no observations may exist yet.
    timeout = float(robot.joints_init_duration or 0.0) + 20.0
    if not robot.ready_event.wait(timeout) or not robot.is_alive():
        raise RuntimeError('Robot did not become ready after joint initialization')


def return_to_initial_joints(env, shm_manager, initial_joints, duration):
    """Reuse the existing worker's joint-init path without competing arm commands."""
    old = env.robots[0]
    print(f'Returning to initial joints over {duration:g}s...', flush=True)
    old.stop(wait=True)  # Includes termination of its Cartesian impedance policy.
    robot = FrankaInterpolationController(
        shm_manager=shm_manager, robot_ip=old.robot_ip,
        robot_port=old.robot_port, frequency=old.frequency,
        joints_init=initial_joints, joints_init_duration=duration,
        receive_latency=old.receive_latency, tx_flange_tcp=old.tx_flange_tcp,
        soft_real_time=old.soft_real_time, verbose=old.verbose,
        get_max_k=old.ring_buffer.get_max_k,
    )
    robot.Kx = old.Kx.copy()
    robot.Kxd = old.Kxd.copy()
    env.robots[0] = robot
    robot.start(wait=False)
    wait_robot_ready(robot)
    time.sleep(0.2)  # Populate observation history before the next get_obs().
    print('Joint return complete. Waiting for C.', flush=True)


def restore_initial_gripper_opening(env, gripper_config):
    """Request the configured absolute opening without changing the motor zero."""
    if 'initial_position' not in gripper_config:
        return
    width = float(gripper_config['initial_position'])
    env.gripper_controller.move_gripper(
        width_m=width,
        force_limit_nm=float(gripper_config.get('force_limit_nm', 1.5)),
    )
    print(f'Initial gripper opening requested: {width * 1000:.1f} mm (zero unchanged).', flush=True)


# ============ 主函数 ============

@click.command()
@click.option('--server-url', '-s', default=None, help='OpenPI WebSocket URL; defaults to policy.server_url in robot config')
@click.option('--prompt', default=None, help='Task prompt; defaults to policy.prompt in robot config')
@click.option('--request-timeout', default=None, type=float, help='WebSocket timeout; defaults to policy.request_timeout_s')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot-config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--camera-reorder', '-cr', default=None, help='Optional comma-separated camera indices')
@click.option('--vis-camera-idx', default=0, type=int, help='Which camera to visualize')
@click.option('--save-policy-data', is_flag=True, help='Save exact request image/state and returned actions; does not disable motion.')
@click.option('--debug-actions', is_flag=True, help='Inspect and log predictions; do not submit policy actions or request homing/initial gripper movement. Controllers still start.')
@click.option('--init-joints', '-j', is_flag=True, default=False, help='Initialize joints once at startup; automatic when YAML joints_init is present')
@click.option('--steps-per-inference', '-si', default=1, type=int, help='Action horizon for inference')
@click.option('--max-duration', '-md', default=2000000, type=float, help='Max duration per episode (seconds)')
@click.option('--max-timesteps', '-mt', default=500, type=int, help='Max steps per episode')
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
         temporal_agg, ensemble_steps, debug_actions, save_policy_data):

    if debug_actions:
        init_joints = False
        print("DEBUG: no policy targets, homing, or initial gripper move. Environment still enables holding controllers and resets gripper zero.")

    # 加载机器人配置
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    policy_config = robot_config_data.get('policy', {})
    server_url = server_url or policy_config.get('server_url')
    prompt = prompt if prompt is not None else policy_config.get('prompt', '')
    request_timeout = request_timeout or float(policy_config.get('request_timeout_s', 30.0))
    frequency = frequency or float(policy_config.get('control_fps', 30.0))
    obs_image_resolution = obs_image_resolution or int(policy_config.get('image_resolution', 240))
    exec_horizon_steps = exec_horizon_steps or int(policy_config.get('exec_horizon_steps', 32))
    action_is_relative = bool(policy_config.get('action_is_relative', True))
    hardware_to_model_gripper_scale = float(
        policy_config.get('hardware_to_model_gripper_scale', 1000.0 / 30.0)
    )
    model_to_hardware_gripper_scale = float(
        policy_config.get('model_to_hardware_gripper_scale', 30.0 / 1000.0)
    )
    dt = 1 / frequency
    obs_res = (obs_image_resolution, obs_image_resolution)
    tx_left_right = np.array(robot_config_data.get('tx_left_right', np.eye(4)))
    tx_robot1_robot0 = tx_left_right

    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    # Fail before hardware startup for unsupported episode-reset configurations.
    if len(robots_config) != 1 or robots_config[0]['robot_type'] != 'franka':
        raise click.UsageError('Episode client currently supports one Franka robot')
    if len(grippers_config) != 1 or grippers_config[0].get('gripper_type') != 'custom' or grippers_config[0].get('mode', 'single') != 'single':
        raise click.UsageError('Episode client requires one custom single-motor gripper')
    if max_timesteps <= 0 or not np.isfinite(max_duration) or max_duration <= 0:
        raise click.UsageError('Episode duration and timestep limits must be positive')
    for rc in robots_config:
        if rc.get('joints_init') is not None:
            joints = np.asarray(rc['joints_init'], dtype=float)
            duration = float(rc.get('joints_init_duration', 5.0))
            if joints.shape != (7,) or not np.all(np.isfinite(joints)) or not np.isfinite(duration) or duration <= 0:
                raise click.UsageError('joints_init requires seven finite values and a positive duration')
    gc = grippers_config[0]
    if 'initial_position' in gc:
        width = float(gc['initial_position'])
        if not np.isfinite(width) or not 0 <= width <= 0.20:
            raise click.UsageError('initial_position must be in metres within [0, 0.20]')
    cameras_config = robot_config_data.get('cameras')
    num_robots = len(robots_config)
    if num_robots != 1:
        raise ValueError("This PI05 adapter expects one robot and a 10D action checkpoint")
    if camera_reorder is None:
        camera_indices = None if cameras_config is not None else [0]
    else:
        camera_indices = [int(x) for x in camera_reorder.split(',')]

    print("=" * 60)
    print("UMI Policy Client (No ROS)")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Number of robots: {num_robots}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Steps per inference: {steps_per_inference}")
    print("=" * 60)

    # 创建策略客户端
    print('[Startup] Connecting to policy server...', flush=True)
    policy_client = PolicyClient(server_url, prompt=prompt, timeout=request_timeout)
    atexit.register(policy_client.close)
    capture_dir = None
    capture_idx = 0
    if save_policy_data:
        import tempfile
        capture_root = os.path.join(os.path.expanduser(output), 'policy_io')
        os.makedirs(capture_root, exist_ok=True)
        capture_dir = tempfile.mkdtemp(prefix='run_', dir=capture_root)
        print(f"Policy request/response files: {capture_dir}")
        with open(os.path.join(capture_dir, 'metadata.json'), 'w') as capture_file:
            json.dump(dict(
                robot_config=robot_config_data, server_metadata=policy_client.metadata,
                server_url=server_url, prompt=prompt, action_is_relative=action_is_relative,
                frequency=frequency, exec_horizon_steps=exec_horizon_steps,
                steps_per_inference=steps_per_inference, debug_actions=debug_actions,
                image_resolution=obs_image_resolution,
                hardware_to_model_gripper_scale=hardware_to_model_gripper_scale,
                model_to_hardware_gripper_scale=model_to_hardware_gripper_scale,
            ), capture_file, indent=2, default=str)
    debug_path = os.path.join(os.path.expanduser(output), 'action_debug.jsonl')
    if debug_actions:
        os.makedirs(os.path.dirname(debug_path), exist_ok=True)
        print(f"Action diagnostics: {debug_path}")
        print(f"Server metadata: {policy_client.metadata}")
        print(f"Configured relative actions: {action_is_relative}; TCP transform: {robots_config[0].get('tx_flange_tcp')}")

    # 设置下采样步数（默认值）
    down_sample_steps = 1

    print('[Startup] Starting keyboard listener and hardware environment...', flush=True)
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=False,
                camera_reorder=camera_indices,
                cameras_config=cameras_config,
                init_joints=(not debug_actions and (
                    init_joints or robots_config[0].get('joints_init') is not None)),
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

            wait_robot_ready(env.robots[0])
            if not debug_actions:
                restore_initial_gripper_opening(env, grippers_config[0])
            rc = robots_config[0]
            initial_joints = np.asarray(
                rc['joints_init'] if rc.get('joints_init') is not None
                else env.robots[0].get_state()['ActualQ'], dtype=float).copy()
            home_duration = float(rc.get('joints_init_duration', 5.0))
            if not np.isfinite(home_duration) or home_duration <= 0:
                raise ValueError('joints_init_duration must be positive and finite')
            print(f'Initial joints for episode return: {initial_joints}')
            print('C starts; S ends; R returns to initial joints and restores gripper opening while idle; Q quits.')
            print('After an episode, submitted actions finish and the arm holds. Homing requires R.')

            print("\n" + "=" * 60)
            print("Ready! Press 'C' to start policy, 'Q' to quit")
            print("=" * 60)

            # 初始化动作数组（用于时序聚合）
            # action_horizon = exec_horizon_steps
            action_dim = 7 * num_robots
            all_time_actions = None
            episode_idx = 0
            # Replay entries may be absent for a stopped/no-action trial. Keep
            # videos separate even when replay_buffer.n_episodes does not grow.
            import tempfile
            trial_video_root = env.video_dir / os.path.basename(
                tempfile.mkdtemp(prefix='trials_', dir=str(env.video_dir)))
            # all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))

            while True:
                # ========= 等待开始 =========
                print("\nWaiting for start command...")
                print('C: start here; R: return to initial joints; Q: quit. Terminal input requires Enter.', flush=True)

                while True:
                    obs = env.get_obs()
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1].copy()

                    # 显示状态
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
                        if not debug_actions:
                            return_to_initial_joints(env, shm_manager, initial_joints, home_duration)
                            restore_initial_gripper_opening(env, grippers_config[0])
                        else:
                            print('Debug: skipping joint return and gripper opening.', flush=True)
                        # Homing never starts inference; require a fresh C afterwards.
                        if 'q' in read_episode_keys(key_counter):
                            return
                        continue
                    if 'c' in press_events:
                        print('Starting episode...', flush=True)
                        break
                    time.sleep(0.05)

                # ========== 策略控制循环 ==========
                quit_requested = False
                last_action_end_time = 0.0
                try:
                    episode_idx += 1
                    env.video_dir = trial_video_root / f'trial_{episode_idx:06d}'
                    env.video_dir.mkdir(parents=True, exist_ok=True)
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

                    # 获取 episode 起始位姿
                    obs = env.get_obs()
                    episode_start_pose = []
                    for robot_id in range(num_robots):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

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

                        # 获取观测
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        image = cv2.resize(
                            obs['camera0_rgb'][-1],
                            (obs_image_resolution, obs_image_resolution),
                            interpolation=cv2.INTER_AREA,
                        ).astype(np.uint8, copy=False)

                        # absolute [x, y, z, rot_vec]
                        current_pose6 = np.concatenate([
                            obs['robot0_eef_pos'][-1],
                            obs['robot0_eef_rot_axis_angle'][-1],
                        ])
                        current_rotmat = st.Rotation.from_rotvec(
                            current_pose6[3:6]
                        ).as_matrix()

                        # absolute [x, y, z, rx1, rx2, rx3, ry1, ry2, ry3]
                        current_pose9 = np.concatenate([
                            current_pose6[:3],
                            current_rotmat[:,:2].T.reshape(6),
                        ])
                        # current_pose9 = pose_6d_to_pose_9d(current_pose6)
                        current_gripper = np.asarray(
                            obs['robot0_gripper_width'][-1], dtype=np.float32
                        ).reshape(-1)[:1] * hardware_to_model_gripper_scale
                        current_state = np.concatenate(
                            [current_pose9, current_gripper]
                        ).astype(np.float32)

                        # Save before inference so failed requests still retain inputs.
                        capture_prefix = None
                        if capture_dir is not None:
                            capture_prefix = os.path.join(capture_dir, f'{capture_idx:06d}')
                            capture_idx += 1
                            np.savez(
                                capture_prefix + '_input.npz',
                                image_rgb=image, state10=current_state,
                                measured_pose6=current_pose6,
                                gripper_width_m=np.asarray(obs['robot0_gripper_width'][-1]),
                                observation_timestamps=np.asarray(obs_timestamps),
                                prompt=np.asarray(prompt), step=np.asarray(iter_idx),
                                episode=np.asarray(episode_idx),
                            )
                            if not cv2.imwrite(capture_prefix + '_image.png', image[..., ::-1]):
                                raise IOError('Could not save policy input image')

                        # 调用服务器获取动作
                        s = time.time()
                        raw_action = policy_client.call_server(image, current_state)
                        response_received_at = time.time()
                        if capture_prefix is not None:
                            np.savez(
                                capture_prefix + '_response.npz',
                                actions=np.asarray(raw_action) if raw_action is not None else np.empty((0, 10), dtype=np.float32),
                                response_valid=np.asarray(raw_action is not None),
                                request_started_at=np.asarray(s),
                                response_received_at=np.asarray(response_received_at),
                            )
                        
                        # print(f"DEBUG: raw_action type: {type(raw_action)}")
                        # print(f"DEBUG: raw_action value: {raw_action}")
                        # if isinstance(raw_action, np.ndarray):
                        #     print(f"DEBUG: raw_action shape: {raw_action.shape}")

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

                        absolute_action10 = (
                            relative_actions_to_absolute_actions(
                                raw_action,
                                base_absolute_action=current_state,
                            )
                            if action_is_relative
                            else raw_action
                        )
                        absolute_rot_mats = ortho6d_to_rotation_matrix(
                            absolute_action10[:, 3:9]
                        )
                        absolute_rotvec = st.Rotation.from_matrix(
                            absolute_rot_mats
                        ).as_rotvec()
                        action = np.concatenate([
                            absolute_action10[:, :3],
                            absolute_rotvec,
                            absolute_action10[:, 9:10]
                            * model_to_hardware_gripper_scale,
                        ], axis=-1)
                        action_horizon = action.shape[0]

                        # all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action
                        if temporal_agg:
                            required_steps = max_timesteps + action_horizon
                            if all_time_actions is None or iter_idx == 0:
                                all_time_actions = np.zeros(
                                    (max_timesteps, required_steps, action_dim)
                                )
                            elif all_time_actions.shape[1] < required_steps:
                                all_time_actions = np.pad(all_time_actions, (
                                    (0, 0),
                                    (0, required_steps - all_time_actions.shape[1]),
                                    (0, 0)))
                            all_time_actions[
                                iter_idx, iter_idx:iter_idx + action_horizon
                            ] = action

                        if inference_idx >= steps_per_inference:
                            inference_idx = 0

                            if temporal_agg:
                                # 时序聚合
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                                target_pose_list = []
                                for i in range(action_horizon):
                                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
                                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                                    if len(actions_for_curr_step) > 0:
                                        k = -0.01
                                        exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                        exp_weights = exp_weights / exp_weights.sum()
                                        weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                        weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                        weighted_action[0][3:6] = weighted_rotvec
                                        target_pose_list.append(weighted_action)
                                    else:
                                        target_pose_list.append(action[[i]])

                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = action

                            # 碰撞检测
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

                            # 处理时序
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
                            
                            # print(f"DEBUG: action_timestamps range: {action_timestamps[0]} to {action_timestamps[-1]}")
                            # print(f"DEBUG: current_time: {curr_time}")
                            # print(f"DEBUG: num_new_actions: {np.sum(is_new)}")

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

                            # 截取中间步骤，丢弃 horizon 末端低质量动作
                            exec_start = 0   # 跳过前几步（可能已过时）
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
                                # Estimate only: observation may be old and this is not
                                # the controller's current commanded/interpolated pose.
                                lead_s = action_timestamps[0] - float(
                                    robots_config[0].get('robot_action_latency', 0.0)
                                ) - time.time()
                                first_target = this_target_poses[0]
                                distance_m = np.linalg.norm(first_target[:3] - current_pose6[:3])
                                angle_rad = (
                                    st.Rotation.from_rotvec(current_pose6[3:6]).inv()
                                    * st.Rotation.from_rotvec(first_target[3:6])
                                ).magnitude()
                                print(
                                    f"[Timing] first target vs request observation: "
                                    f"distance={distance_m * 1000:.1f}mm "
                                    f"rotation={np.rad2deg(angle_rad):.2f}deg "
                                    f"compensated_lead={lead_s * 1000:.1f}ms; "
                                    + (f"distance/time estimate={distance_m / lead_s:.3f}m/s, "
                                       f"{np.rad2deg(angle_rad) / lead_s:.1f}deg/s "
                                       "(not measured velocity)" if lead_s > 0 else "TARGET ALREADY DUE")
                                )


                        #    # 执行动作 (pdb单步调试: 按c继续执行, 按q退出)
                        #     print(f"\n[PDB] Step {iter_idx}: About to execute {len(this_target_poses)} actions")
                        #     print(f"[PDB] Target poses:\n{this_target_poses}")
                        #     breakpoint()

                        #     # pdb恢复后重新计算时间戳，确保在未来
                        #     action_timestamps = (np.arange(len(this_target_poses), dtype=np.float64)) * dt + time.time() + 0.01

                            if debug_actions:
                                # Observe only: never send a HOLD or model command here.
                                latest = env.get_obs()
                                measured = np.concatenate([
                                    latest['robot0_eef_pos'][-1],
                                    latest['robot0_eef_rot_axis_angle'][-1],
                                ])
                                checked_at = time.time()
                                robot_latency = float(robots_config[0].get('robot_action_latency', 0.0))
                                lead_ms = (action_timestamps - robot_latency - checked_at) * 1000
                                translation_mm = np.linalg.norm(this_target_poses[:, :3] - measured[:3], axis=1) * 1000
                                rotation_deg = np.rad2deg((
                                    st.Rotation.from_rotvec(measured[3:6]).inv()
                                    * st.Rotation.from_rotvec(this_target_poses[:, 3:6])
                                ).magnitude())
                                record = dict(
                                    episode=episode_idx, step=iter_idx, wall_time=checked_at,
                                    server_metadata=policy_client.metadata,
                                    configured_relative=action_is_relative,
                                    tx_flange_tcp=robots_config[0].get('tx_flange_tcp'),
                                    hardware_to_model_gripper_scale=hardware_to_model_gripper_scale,
                                    model_to_hardware_gripper_scale=model_to_hardware_gripper_scale,
                                    input_state10=current_state.tolist(),
                                    inference_base_pose6=current_pose6.tolist(),
                                    image_shape=list(image.shape), image_dtype=str(image.dtype),
                                    raw_actions10=np.asarray(raw_action).tolist(),
                                    decoded_absolute_actions7=action.tolist(),
                                    selected_targets7=this_target_poses.tolist(),
                                    measured_pose6=measured.tolist(),
                                    measured_gripper_m=np.asarray(latest['robot0_gripper_width'][-1]).tolist(),
                                    observation_timestamp=float(obs_timestamps[-1]),
                                    latest_observation_timestamp=float(latest['timestamp'][-1]),
                                    network_latency_s=inference_time,
                                    action_timestamps=action_timestamps.tolist(),
                                    robot_action_latency_s=robot_latency,
                                    compensated_lead_ms=lead_ms.tolist(),
                                    translation_delta_mm=translation_mm.tolist(),
                                    rotation_delta_deg=rotation_deg.tolist(),
                                    targets_submitted=False,
                                )
                                with open(debug_path, 'a') as debug_file:
                                    debug_file.write(json.dumps(record, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + "\n")
                                print(f"DEBUG step {iter_idx}: NO TARGETS SUBMITTED")
                                print(f"  measured xyz(m)/rotvec(rad): {measured}")
                                print(f"  raw first action: {raw_action[0]}")
                                for j, target in enumerate(this_target_poses[:3]):
                                    print(f"  target[{j}] xyz(m)/rotvec(rad)/gripper(m): {target}")
                                    print(f"    delta={translation_mm[j]:.2f} mm, {rotation_deg[j]:.2f} deg; compensated lead={lead_ms[j]:.1f} ms")
                            else:
                                # Track the entire submitted chunk, including a
                                # possible partial submission if enqueue fails.
                                last_action_end_time = max(
                                    last_action_end_time, float(np.max(action_timestamps)))
                                env.exec_actions(
                                    actions=this_target_poses,
                                    timestamps=action_timestamps,
                                    compensate_latency=True
                                )
                                print(f"Step {iter_idx}: Submitted {len(this_target_poses)} actions, gripper={action[0, 6]:.4f}")
                            # print(f"Step {iter_idx}: Submitted {len(this_target_poses)} actions")
                        # 可视化
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
                        if not debug_actions:
                            print('[Episode] Waiting for submitted robot actions to finish...', flush=True)
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

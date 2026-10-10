"""
UMI Policy Client - 策略推理客户端（无 ROS 依赖）

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
"""
import os
import sys
import time
import json
import yaml
import click
import cv2
import numpy as np
import requests
import scipy.spatial.transform as st
from multiprocessing.managers import SharedMemoryManager

# 限制线程数
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import KeystrokeCounter, KeyCode
from umi.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
from scipy.spatial.transform import Rotation as R

# # 尝试导入 json_numpy
# json_numpy = None
# USE_JSON_NUMPY = True
try:
    import json_numpy as _json_numpy
    _json_numpy.patch()
    json_numpy = _json_numpy
    USE_JSON_NUMPY = True
except ImportError:
    print("Warning: json_numpy not found, using standard json")


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
    """策略客户端 - 与远程服务器通信并控制机器人"""

    def __init__(self, server_url: str):
        if not server_url.startswith('http'):
            server_url = f'http://{server_url}'
        self.server_url = server_url
        self.session = requests.Session()

    def call_server(self, obs_dict: dict) -> np.ndarray:
        """
        发送观测到服务器并获取动作

        Args:
            obs_dict: 观测字典，包含图像和状态

        Returns:
            np.ndarray: 动作数组 (horizon, action_dim)
        """
        global json_numpy
        # 发送请求
        if USE_JSON_NUMPY and json_numpy is not None:
            body = json_numpy.dumps(obs_dict)
        else:
            # 手动转换 numpy 数组
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(v) for v in obj]
                else:
                    return obj
            payload = convert_to_serializable(obs_dict)
            body = json.dumps(payload)

        try:
            resp = self.session.post(
                f"{self.server_url}/dp",
                data=body,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            resp.raise_for_status()
            
            # --- 修改开始 ---
            if USE_JSON_NUMPY and json_numpy is not None:
                # 使用 json_numpy 解析，它会自动识别 {"__numpy__": ...} 并还原成真正的 ndarray
                result = json_numpy.loads(resp.text)
            else:
                result = json.loads(resp.text)
            
            # 确保获取的是 action 键
            action = result["action"]
            
            # 关键一步：如果服务器返回的是经过序列化的字典，而 loads 没能自动转换
            if isinstance(action, dict) and "__numpy__" in action:
                import json_numpy
                action = json_numpy.decode(action)
                
            return np.array(action)
            # --- 修改结束 ---
            
            # result = json.loads(resp.text)
            # return np.array(result["action"])
        except Exception as e:
            print(f"[PolicyClient] Server error: {e}")
            return None


# ============ 主函数 ============

@click.command()
@click.option('--server-url', '-s', required=True, help='Policy server URL (e.g., http://192.168.1.100:8099)')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot-config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--camera-reorder', '-cr', default='0', help='Camera reorder indices')
@click.option('--vis-camera-idx', default=0, type=int, help='Which camera to visualize')
@click.option('--init-joints', '-j', is_flag=True, default=False, help='Initialize robot to home position')
@click.option('--steps-per-inference', '-si', default=1, type=int, help='Action horizon for inference')
@click.option('--max-duration', '-md', default=2000000, type=float, help='Max duration per episode (seconds)')
@click.option('--max-timesteps', '-mt', default=500, type=int, help='Max steps per episode')
@click.option('--frequency', '-f', default=5, type=float, help='Control frequency (Hz)')
@click.option('--obs-image-resolution', default=224, type=int, help='Observation image resolution')
@click.option('--no-mirror', '-nm', is_flag=True, default=False, help='Disable mirror')
@click.option('--mirror-swap', is_flag=True, default=False, help='Enable mirror swap')
@click.option('--temporal-agg', is_flag=True, default=False, help='Enable temporal aggregation')
@click.option('--ensemble-steps', type=int, default=8, help='Ensemble steps for temporal agg')
def main(server_url, output, robot_config,
         camera_reorder, vis_camera_idx, init_joints,
         steps_per_inference, max_duration, max_timesteps,
         frequency, obs_image_resolution, no_mirror, mirror_swap,
         temporal_agg, ensemble_steps):

    max_gripper_width = 0.09
    dt = 1 / frequency
    obs_res = (obs_image_resolution, obs_image_resolution)

    # 加载机器人配置
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    tx_left_right = np.array(robot_config_data.get('tx_left_right', np.eye(4)))
    tx_robot1_robot0 = tx_left_right
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']
    num_robots = len(robots_config)

    print("=" * 60)
    print("UMI Policy Client (No ROS)")
    print("=" * 60)
    print(f"Server URL: {server_url}")
    print(f"Number of robots: {num_robots}")
    print(f"Control frequency: {frequency} Hz")
    print(f"Steps per inference: {steps_per_inference}")
    print("=" * 60)

    # 创建策略客户端
    policy_client = PolicyClient(server_url)

    # 设置下采样步数（默认值）
    down_sample_steps = 1

    print("Starting environment...")
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=False,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
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

            # ========== 机器人初始化 ==========
            if init_joints:
                print("Initializing robot to home position using servoL...")
                try:
                    # Wait for robot state buffer to fill
                    time.sleep(0.5)

                    # Get current pose for reference1.426010, 1.571342, -1.179061
                    obs = env.get_obs()
                    current_pos = obs['robot0_eef_pos'][-1]
                    current_rot = obs['robot0_eef_rot_axis_angle'][-1]
                    current_pose = np.concatenate([current_pos, current_rot])
                    print(f"Current EEF pose: {current_pose}")

                    # Define target home pose - ADJUST THESE VALUES FOTOTAL=125.3723msR YOUR SETUP!
                    # Position: [x, y, z] in meters
                    # Rotation: axis-angle representation
                    # target_home_pose = np.array([
                    #     0.461525, 0.012424, 0.253867,      # x, y, z position1.426010, 1.571342, -1.179061
                    #     -1.581865, 1.532763, -0.887445     # axis-angle rotation
                    # ])
                    # target_home_pose = np.array([
                    #     0.467959, -0.004394, 0.154206,      # x, y, z position
                    #     -1.663930, 1.561844, -0.925636     # axis-angle rotation
                    # ])
                    
                    # target_home_pose = np.array([
                    #      0.497524, 0.037813, 0.212450,      # x, y, z position
                    #     -1.450700, 1.597360, -1.181150     # axis-angle rotation
                    # ])
                    
                    # target_home_pose = np.array([
                    #      0.555839, -0.081934, 0.221995,      # x, y, z position
                    #     -2.116637, 0.819526, -0.431101     # axis-angle rotation
                    # ])
                    
                    target_home_pose = np.array([
                         0.508993, -0.078624, 0.132818,      # x, y, z position
                        -2.732883, 0.632504, -0.124448     # axis-angle rotation
                    ])
                    
                    print(f"Target home pose: {target_home_pose}")

                    # Move robot using servoL
                    init_duration = 3.0
                    env.robots[0].servoL(target_home_pose, duration=init_duration)
                    print(f"Moving to home position (duration: {init_duration}s)...")
                    time.sleep(init_duration + 0.5)

                    # Open gripper
                    print("Opening gripper...")
                    gripper_ctrl = env.gripper_controller
                    if hasattr(gripper_ctrl, 'move_gripper'):
                        gripper_ctrl.move_gripper(width_m=0.035, force_limit_nm=1.5)
                    elif hasattr(gripper_ctrl, 'schedule_waypoint'):
                        gripper_ctrl.schedule_waypoint(0.08, target_time=time.time() + 1.0)
                    time.sleep(1.5)

                    print("Robot initialization complete!")
                except Exception as e:
                    print(f"[Warning] Initialization failed: {e}")
                    import traceback
                    traceback.print_exc()

            print("\n" + "=" * 60)
            print("Ready! Press 'C' to start policy, 'Q' to quit")
            print("=" * 60)

            # 初始化动作数组（用于时序聚合）
            action_horizon = 16  # 默认值
            action_dim = 7 * num_robots
            all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))

            while True:
                # ========= 等待开始 =========
                print("\nWaiting for start command...")
                print("Press 'C' to start policy control, 'Q' to quit")

                while True:
                    obs = env.get_obs()
                    vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]

                    # 显示状态
                    text = 'Press C to start, Q to quit'
                    cv2.putText(vis_img, text, (10, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.5, thickness=1, color=(255, 255, 255))
                    cv2.imshow('default', vis_img[..., ::-1])
                    cv2.pollKey()

                    press_events = key_counter.get_press_events()
                    start_policy = False
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            print("Exiting...")
                            env.end_episode()
                            return
                        elif key_stroke == KeyCode(char='c'):
                            start_policy = True
                            break

                    if start_policy:
                        break
                    time.sleep(0.05)

                # ========== 策略控制循环 ==========
                try:
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
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # 获取观测
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency: {time.time() - obs_timestamps[-1]:.3f}s')

                        # 构建发送给服务器的观测字典 - 使用服务器期望的格式
                        # 服务器期望: fisheye_img, left_robot_tcp_xyzrpy (xyz+rpy), left_robot_gripper_width_for_action
                        eef_pos = obs['robot0_eef_pos']  # (T, 3)
                        eef_rot_aa = obs['robot0_eef_rot_axis_angle']  # (T, 3) axis-angle
                        eef_rot_rpy = axis_angle_to_rpy(eef_rot_aa)  # 转换为 RPY
                        xyzrpy = np.concatenate([eef_pos, eef_rot_rpy], axis=-1)  # (T, 6)

                        server_obs = {
                            'fisheye_img': obs['camera0_rgb'],  # (T, H, W, C)
                            'left_robot_tcp_xyzrpy': xyzrpy,  # (T, 6) xyz + rpy
                            'left_robot_gripper_width_for_action': obs['robot0_gripper_width'],  # (T, 1)
                            'episode_start_pose': episode_start_pose,
                            'tx_robot1_robot0': tx_robot1_robot0.tolist() if isinstance(tx_robot1_robot0, np.ndarray) else tx_robot1_robot0,
                        }

                        # 如果是双臂，添加第二个机器人的观测
                        if num_robots > 1:
                            eef_pos1 = obs['robot1_eef_pos']
                            eef_rot_aa1 = obs['robot1_eef_rot_axis_angle']
                            eef_rot_rpy1 = axis_angle_to_rpy(eef_rot_aa1)
                            xyzrpy1 = np.concatenate([eef_pos1, eef_rot_rpy1], axis=-1)
                            server_obs['right_robot_tcp_xyzrpy'] = xyzrpy1
                            server_obs['right_robot_gripper_width_for_action'] = obs['robot1_gripper_width']

                        # 调用服务器获取动作
                        s = time.time()
                        raw_action = policy_client.call_server(server_obs)
                        
                        # print(f"DEBUG: raw_action type: {type(raw_action)}")
                        # print(f"DEBUG: raw_action value: {raw_action}")
                        # if isinstance(raw_action, np.ndarray):
                        #     print(f"DEBUG: raw_action shape: {raw_action.shape}")

                        if raw_action is None or np.array(raw_action).size == 0:
                            print("Warning: Received empty action from server!")
                            continue
                        
                        inference_time = time.time() - s
                        print(f'Network latency: {inference_time:.3f}s')

                        if raw_action is None:
                            print("No action from server, skipping...")
                            precise_wait(t_cycle_end - frame_latency)
                            iter_idx += 1
                            continue

                        # 服务器返回格式: xyz(3) + rpy(3) + gripper(1) per robot
                        # 需要转换为: xyz(3) + axis_angle(3) + gripper(1) per robot
                        action_dim_per_robot = 7  # xyz(3) + rot(3) + gripper(1)
                        converted_actions = []
                        for robot_idx in range(num_robots):
                            start = robot_idx * action_dim_per_robot
                            xyz = raw_action[..., start:start+3]
                            rpy = raw_action[..., start+3:start+6]
                            gripper = raw_action[..., start+6:start+7]
                            # RPY -> axis-angle
                            axis_angle = rpy_to_axis_angle(rpy)
                            robot_action = np.concatenate([xyz, axis_angle, gripper], axis=-1)
                            converted_actions.append(robot_action)
                        action = np.concatenate(converted_actions, axis=-1)  # (horizon, action_dim)
                        action_horizon = action.shape[0]
                        all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

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
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            
                            # print(f"DEBUG: action_timestamps range: {action_timestamps[0]} to {action_timestamps[-1]}")
                            # print(f"DEBUG: current_time: {curr_time}")
                            # print(f"DEBUG: num_new_actions: {np.sum(is_new)}")

                            if np.sum(is_new) == 0:
                                this_target_poses = this_target_poses[[-1]]
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(f'Over budget: {action_timestamp - curr_time:.3f}s')
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # 截取中间步骤，丢弃 horizon 末端低质量动作
                            exec_start = 0   # 跳过前几步（可能已过时）
                            exec_end = 16    # 丢弃末端反向动作
                            this_target_poses = this_target_poses[exec_start:exec_end]
                            action_timestamps = action_timestamps[exec_start:exec_end]


                        #    # 执行动作 (pdb单步调试: 按c继续执行, 按q退出)
                        #     print(f"\n[PDB] Step {iter_idx}: About to execute {len(this_target_poses)} actions")
                        #     print(f"[PDB] Target poses:\n{this_target_poses}")
                        #     breakpoint()

                        #     # pdb恢复后重新计算时间戳，确保在未来
                        #     action_timestamps = (np.arange(len(this_target_poses), dtype=np.float64)) * dt + time.time() + 0.01

                            # 单独发送夹爪指令（取截断前 action 的第0步，不受 exec_start 影响）
                            gripper_target = action[3, 6]  # 第0步的 gripper width
                            env.gripper_controller.move_gripper(gripper_target, force_limit_nm=1.5)

                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            print(f"Step {iter_idx}: Submitted {len(this_target_poses)} actions, gripper={gripper_target:.4f}")
                            # print(f"Step {iter_idx}: Submitted {len(this_target_poses)} actions")
                        # 可视化
                        vis_img = obs[f'camera{vis_camera_idx}_rgb'][-1]
                        text = f'Step: {iter_idx}, Time: {time.monotonic() - t_start:.1f}s'
                        cv2.putText(vis_img, text, (10, 20),
                                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.5, thickness=1, color=(255, 255, 255))
                        cv2.imshow('default', vis_img[..., ::-1])
                        cv2.pollKey()

                        # 检查停止
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                print('Stop requested.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max duration reached.")
                            stop_episode = True
                        if iter_idx + 1 >= max_timesteps:
                            print("Max timesteps reached.")
                            stop_episode = True

                        if stop_episode:
                            env.end_episode()
                            break

                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("\nInterrupted!")
                    env.end_episode()

                print("Episode ended.")


if __name__ == '__main__':
    main()

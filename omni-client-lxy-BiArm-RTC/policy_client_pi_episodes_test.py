r"""
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


class TimeAlignedActionEnsembler:
    """Fuse absolute 7D action chunks that predict the same 30 Hz tick."""

    def __init__(
            self, dt, time_origin, max_chunks=3,
            consensus_translation_m=0.02,
            consensus_rotation_rad=np.deg2rad(5.0),
            consensus_gripper_m=0.01):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be positive and finite')
        if not np.isfinite(time_origin):
            raise ValueError('time_origin must be finite')
        if max_chunks <= 0:
            raise ValueError('max_chunks must be positive')
        if (consensus_translation_m <= 0 or consensus_rotation_rad <= 0
                or consensus_gripper_m <= 0):
            raise ValueError('fusion consensus thresholds must be positive')
        self.dt = float(dt)
        self.time_origin = float(time_origin)
        self.max_chunks = int(max_chunks)
        self.consensus_translation_m = float(consensus_translation_m)
        self.consensus_rotation_rad = float(consensus_rotation_rad)
        self.consensus_gripper_m = float(consensus_gripper_m)
        self._chunks = []
        self.last_candidate_counts = np.empty((0,), dtype=np.int64)

    def _timestamps_to_ticks(self, timestamps):
        return np.rint(
            (np.asarray(timestamps, dtype=np.float64) - self.time_origin) / self.dt
        ).astype(np.int64)

    def _ticks_to_timestamps(self, ticks):
        return self.time_origin + np.asarray(ticks, dtype=np.float64) * self.dt

    def _add_chunk_with_timestamps(self, actions, timestamps):
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
            raise ValueError(f'Expected non-empty actions shaped [T, 7], got {actions.shape}')
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if timestamps.shape != (len(actions),):
            raise ValueError('Each action must have one timestamp')
        if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(timestamps)):
            raise ValueError('Actions and timestamps must be finite')
        self._chunks.append({
            'ticks': self._timestamps_to_ticks(timestamps),
            'actions': actions.copy(),
        })
        if len(self._chunks) > self.max_chunks:
            self._chunks = self._chunks[-self.max_chunks:]

    def add_chunk(self, actions, observation_timestamp):
        """Add a chunk on its original observation-relative timeline."""
        actions = np.asarray(actions)
        timestamps = float(observation_timestamp) + (
            np.arange(1, len(actions) + 1, dtype=np.float64) * self.dt
        )
        self._add_chunk_with_timestamps(actions, timestamps)

    def add_retimed_chunk(self, actions, first_action_timestamp):
        """Add a chunk whose first frame starts on a new execution timeline."""
        actions = np.asarray(actions)
        timestamps = float(first_action_timestamp) + (
            np.arange(len(actions), dtype=np.float64) * self.dt
        )
        self._add_chunk_with_timestamps(actions, timestamps)

    def reject_inconsistent_chunks(
            self, measured_pose6, observation_timestamp,
            max_translation_m, max_rotation_rad):
        """Drop old chunks whose prediction for this observation was not tracked."""
        measured_pose6 = np.asarray(measured_pose6, dtype=np.float64)
        if measured_pose6.shape != (6,) or not np.all(np.isfinite(measured_pose6)):
            raise ValueError('measured_pose6 must contain six finite values')
        if max_translation_m <= 0 or max_rotation_rad <= 0:
            raise ValueError('consistency thresholds must be positive')

        observation_tick = self._timestamps_to_ticks([observation_timestamp])[0]
        measured_rotation = R.from_rotvec(measured_pose6[3:6])
        kept_chunks = []
        rejected = []
        for chunk in self._chunks:
            matches = np.flatnonzero(chunk['ticks'] == observation_tick)
            if len(matches) == 0:
                kept_chunks.append(chunk)
                continue
            predicted = chunk['actions'][matches[0]]
            translation_m = float(np.linalg.norm(predicted[:3] - measured_pose6[:3]))
            rotation_rad = float(
                (measured_rotation.inv() * R.from_rotvec(predicted[3:6])).magnitude()
            )
            if translation_m > max_translation_m or rotation_rad > max_rotation_rad:
                rejected.append((translation_m, rotation_rad))
            else:
                kept_chunks.append(chunk)
        self._chunks = kept_chunks
        return rejected

    def prune(self, before_time):
        """Remove fully expired chunks while retaining history for tracking checks."""
        self._chunks = [
            chunk for chunk in self._chunks
            if np.any(self._ticks_to_timestamps(chunk['ticks']) > before_time)
        ]

    def get_fused_trajectory(self, after_time, max_steps):
        if max_steps <= 0:
            raise ValueError('max_steps must be positive')
        self.prune(after_time)
        if not self._chunks:
            self.last_candidate_counts = np.empty((0,), dtype=np.int64)
            return (
                np.empty((0, 7), dtype=np.float64),
                np.empty((0,), dtype=np.float64),
                np.empty((0,), dtype=np.int64),
            )

        ticks = np.unique(np.concatenate([chunk['ticks'] for chunk in self._chunks]))
        ticks = ticks[self._ticks_to_timestamps(ticks) > after_time]
        ticks = ticks[:max_steps]
        fused_actions = []
        contributor_counts = []
        candidate_counts = []
        for tick in ticks:
            contributors = np.stack([
                chunk['actions'][np.flatnonzero(chunk['ticks'] == tick)[0]]
                for chunk in self._chunks
                if np.any(chunk['ticks'] == tick)
            ])
            # The latest chunk defines the current task phase. Do not mix an
            # older reach/close/return phase into a newer phase merely because
            # both predict the same wall-clock tick.
            newest = contributors[-1]
            translation_error = np.linalg.norm(
                contributors[:, :3] - newest[:3], axis=1
            )
            rotation_error = (
                R.from_rotvec(newest[3:6]).inv()
                * R.from_rotvec(contributors[:, 3:6])
            ).magnitude()
            gripper_error = np.abs(contributors[:, 6] - newest[6])
            same_phase = (
                (translation_error <= self.consensus_translation_m)
                & (rotation_error <= self.consensus_rotation_rad)
                & (gripper_error <= self.consensus_gripper_m)
            )
            accepted = contributors[same_phase]
            fused = accepted.mean(axis=0)
            fused[3:6] = R.from_rotvec(accepted[:, 3:6]).mean().as_rotvec()
            # Gripper phase transitions must never be advanced by an old chunk.
            fused[6] = newest[6]
            fused_actions.append(fused)
            contributor_counts.append(len(accepted))
            candidate_counts.append(len(contributors))

        self.last_candidate_counts = np.asarray(candidate_counts, dtype=np.int64)

        return (
            np.asarray(fused_actions, dtype=np.float64),
            self._ticks_to_timestamps(ticks),
            np.asarray(contributor_counts, dtype=np.int64),
        )

    @property
    def chunk_count(self):
        return len(self._chunks)

    def clear(self):
        self._chunks.clear()
        self.last_candidate_counts = np.empty((0,), dtype=np.int64)


class SubmittedActionTimeline:
    """Remember targets already queued on the controller's physical timeline."""

    def __init__(self, dt, time_origin):
        self.dt = float(dt)
        self.time_origin = float(time_origin)
        self._actions_by_tick = {}

    def _timestamp_to_tick(self, timestamp):
        return int(np.rint((float(timestamp) - self.time_origin) / self.dt))

    def update(self, actions, timestamps):
        actions = np.asarray(actions, dtype=np.float64)
        timestamps = np.asarray(timestamps, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != 7:
            raise ValueError('submitted actions must have shape [T, 7]')
        if timestamps.shape != (len(actions),):
            raise ValueError('submitted actions and timestamps must have equal length')
        for action, timestamp in zip(actions, timestamps):
            self._actions_by_tick[self._timestamp_to_tick(timestamp)] = action.copy()

    def get(self, timestamp):
        action = self._actions_by_tick.get(self._timestamp_to_tick(timestamp))
        return None if action is None else action.copy()

    def prune(self, before_time):
        before_tick = self._timestamp_to_tick(before_time)
        self._actions_by_tick = {
            tick: action for tick, action in self._actions_by_tick.items()
            if tick >= before_tick
        }


def reanchor_action_chunk(actions, anchor_pose6):
    """Move an absolute chunk's first TCP pose to a measured or committed pose."""
    actions = np.asarray(actions, dtype=np.float64)
    anchor_pose6 = np.asarray(anchor_pose6, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError('actions must be non-empty with shape [T, 7]')
    if anchor_pose6.shape != (6,) or not np.all(np.isfinite(anchor_pose6)):
        raise ValueError('anchor_pose6 must contain six finite values')

    predicted_mats = pose_to_mat(actions[:, :6])
    anchor_mat = pose_to_mat(anchor_pose6)
    tx_anchor_predicted0 = anchor_mat @ np.linalg.inv(predicted_mats[0])
    result = actions.copy()
    result[:, :6] = mat_to_pose(tx_anchor_predicted0 @ predicted_mats)
    return result


def compute_latency_trimmed_steps(
        observation_timestamp, planning_time, execution_margin, dt, horizon):
    """Return model frames that are already stale at the first executable tick."""
    values = np.asarray([
        observation_timestamp, planning_time, execution_margin, dt,
    ], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError('latency-trim timing values must be finite')
    if execution_margin < 0 or dt <= 0:
        raise ValueError('execution_margin must be non-negative and dt positive')
    if horizon <= 0:
        raise ValueError('action horizon must be positive')

    elapsed_steps = np.floor(
        (planning_time + execution_margin - observation_timestamp) / dt
    )
    return int(np.clip(elapsed_steps, 0, horizon - 1))


def check_trajectory_safety(
        actions, timestamps, measured_pose6, checked_at, robot_action_latency,
        max_first_translation_m, max_first_rotation_rad,
        max_translation_speed_m_s, max_rotation_speed_rad_s):
    """Validate a scheduled Cartesian trajectory before it reaches hardware."""
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    measured_pose6 = np.asarray(measured_pose6, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) == 0:
        raise ValueError('actions must be non-empty with shape [T, 7]')
    if timestamps.shape != (len(actions),) or measured_pose6.shape != (6,):
        raise ValueError('timestamps or measured_pose6 has an invalid shape')

    first_translation_m = float(np.linalg.norm(actions[0, :3] - measured_pose6[:3]))
    first_rotation_rad = float((
        R.from_rotvec(measured_pose6[3:6]).inv()
        * R.from_rotvec(actions[0, 3:6])
    ).magnitude())
    compensated_timestamps = timestamps - float(robot_action_latency)
    segment_dt = np.diff(np.concatenate([[float(checked_at)], compensated_timestamps]))
    positions = np.vstack([measured_pose6[:3], actions[:, :3]])
    segment_translation_m = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    rotations = R.from_rotvec(np.vstack([measured_pose6[3:6], actions[:, 3:6]]))
    segment_rotation_rad = (rotations[:-1].inv() * rotations[1:]).magnitude()

    valid_dt = segment_dt > 0
    translation_speed = np.full(len(actions), np.inf, dtype=np.float64)
    rotation_speed = np.full(len(actions), np.inf, dtype=np.float64)
    translation_speed[valid_dt] = segment_translation_m[valid_dt] / segment_dt[valid_dt]
    rotation_speed[valid_dt] = segment_rotation_rad[valid_dt] / segment_dt[valid_dt]

    reasons = []
    if not np.all(valid_dt):
        reasons.append('one or more targets are already due')
    if first_translation_m > max_first_translation_m:
        reasons.append(
            f'first translation {first_translation_m * 1000:.1f}mm > '
            f'{max_first_translation_m * 1000:.1f}mm')
    if first_rotation_rad > max_first_rotation_rad:
        reasons.append(
            f'first rotation {np.rad2deg(first_rotation_rad):.2f}deg > '
            f'{np.rad2deg(max_first_rotation_rad):.2f}deg')
    max_translation_speed = float(np.max(translation_speed))
    max_rotation_speed = float(np.max(rotation_speed))
    if max_translation_speed > max_translation_speed_m_s:
        reasons.append(
            f'translation speed {max_translation_speed:.3f}m/s > '
            f'{max_translation_speed_m_s:.3f}m/s')
    if max_rotation_speed > max_rotation_speed_rad_s:
        reasons.append(
            f'rotation speed {np.rad2deg(max_rotation_speed):.1f}deg/s > '
            f'{np.rad2deg(max_rotation_speed_rad_s):.1f}deg/s')
    return {
        'passed': not reasons,
        'reasons': reasons,
        'first_translation_m': first_translation_m,
        'first_rotation_rad': first_rotation_rad,
        'max_translation_speed_m_s': max_translation_speed,
        'max_rotation_speed_rad_s': max_rotation_speed,
    }


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
@click.option(
    '--steps-per-inference', '-si', default=4, type=int,
    help='Inference responses collected between trajectory submissions',
)
@click.option('--max-duration', '-md', default=2000000, type=float, help='Max duration per episode (seconds)')
@click.option('--max-timesteps', '-mt', default=500, type=int, help='Max steps per episode')
@click.option('--frequency', '-f', default=None, type=float, help='Control frequency; defaults to policy.control_fps')
@click.option('--obs-image-resolution', default=None, type=int, help='Image size; defaults to policy.image_resolution')
@click.option('--exec-horizon-steps', default=None, type=int, help='Actions scheduled per inference; defaults to policy.exec_horizon_steps')
@click.option('--no-mirror', '-nm', is_flag=True, default=False, help='Disable mirror')
@click.option('--mirror-swap', is_flag=True, default=False, help='Enable mirror swap')
@click.option('--temporal-agg/--no-temporal-agg', default=True,
              help='Fuse overlapping action chunks on their absolute 30 Hz timestamps')
@click.option('--ensemble-steps', type=int, default=3,
              help='Maximum recent action chunks contributing to temporal aggregation')
@click.option('--reanchor-action-chunks/--preserve-observation-timing', default=False,
              help='Experimental: restart returned chunks at execution time; default preserves model-frame timing')
@click.option('--fusion-consensus-pos-mm', type=float, default=20.0,
              help='Exclude old same-tick predictions farther from the newest prediction')
@click.option('--fusion-consensus-rot-deg', type=float, default=5.0,
              help='Exclude old same-tick predictions with a different rotation phase')
@click.option('--fusion-consensus-gripper-mm', type=float, default=10.0,
              help='Exclude old same-tick predictions with a different gripper phase')
@click.option('--ensemble-consistency-pos-mm', type=float, default=20.0,
              help='Drop an old chunk when its current-position error exceeds this value')
@click.option('--ensemble-consistency-rot-deg', type=float, default=5.0,
              help='Drop an old chunk when its current-rotation error exceeds this value')
@click.option('--max-first-target-pos-mm', type=float, default=20.0,
              help='Reject submission when its first target is farther than this value')
@click.option('--max-first-target-rot-deg', type=float, default=5.0,
              help='Reject submission when its first target rotation exceeds this value')
@click.option('--max-target-speed-m-s', type=float, default=0.5,
              help='Reject submission when any scheduled translation segment is faster')
@click.option('--max-target-rot-speed-deg-s', type=float, default=120.0,
              help='Reject submission when any scheduled rotation segment is faster')
def main(server_url, prompt, request_timeout, output, robot_config,
         camera_reorder, vis_camera_idx, init_joints,
         steps_per_inference, max_duration, max_timesteps,
         frequency, obs_image_resolution, exec_horizon_steps, no_mirror, mirror_swap,
         temporal_agg, ensemble_steps, reanchor_action_chunks,
         fusion_consensus_pos_mm, fusion_consensus_rot_deg,
         fusion_consensus_gripper_mm,
         ensemble_consistency_pos_mm,
         ensemble_consistency_rot_deg, max_first_target_pos_mm,
         max_first_target_rot_deg, max_target_speed_m_s,
         max_target_rot_speed_deg_s, debug_actions, save_policy_data):

    if debug_actions:
        init_joints = False
        print(
            'DEBUG: no policy targets, homing, or initial gripper move. '
            'The policy receives the configured initial gripper opening; '
            'diagnostics still record the measured opening.'
        )

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
    if steps_per_inference <= 0:
        raise click.UsageError('steps_per_inference must be positive')
    if ensemble_steps <= 0:
        raise click.UsageError('ensemble_steps must be positive')
    safety_values = {
        'ensemble_consistency_pos_mm': ensemble_consistency_pos_mm,
        'ensemble_consistency_rot_deg': ensemble_consistency_rot_deg,
        'fusion_consensus_pos_mm': fusion_consensus_pos_mm,
        'fusion_consensus_rot_deg': fusion_consensus_rot_deg,
        'fusion_consensus_gripper_mm': fusion_consensus_gripper_mm,
        'max_first_target_pos_mm': max_first_target_pos_mm,
        'max_first_target_rot_deg': max_first_target_rot_deg,
        'max_target_speed_m_s': max_target_speed_m_s,
        'max_target_rot_speed_deg_s': max_target_rot_speed_deg_s,
    }
    if any(not np.isfinite(value) or value <= 0 for value in safety_values.values()):
        raise click.UsageError('All ensemble consistency and motion safety limits must be positive and finite')
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
    print(f"Time-aligned action ensemble: {temporal_agg} (max chunks: {ensemble_steps})")
    print(f"Re-anchor and retime returned chunks: {reanchor_action_chunks}")
    print(
        'Phase-consensus fusion: '
        f'{fusion_consensus_pos_mm:g}mm/'
        f'{fusion_consensus_rot_deg:g}deg/'
        f'{fusion_consensus_gripper_mm:g}mm gripper; '
        'gripper always comes from the newest chunk'
    )
    print(
        'Safety limits: '
        f'consistency={ensemble_consistency_pos_mm:g}mm/'
        f'{ensemble_consistency_rot_deg:g}deg, '
        f'first target={max_first_target_pos_mm:g}mm/'
        f'{max_first_target_rot_deg:g}deg, '
        f'speed={max_target_speed_m_s:g}m/s/'
        f'{max_target_rot_speed_deg_s:g}deg/s'
    )
    print(
        f'Trajectory submission cadence: every {steps_per_inference} '
        'inference response(s); the first response is submitted immediately.'
    )
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
                temporal_agg=temporal_agg, ensemble_steps=ensemble_steps,
                reanchor_action_chunks=reanchor_action_chunks,
                fusion_consensus_pos_mm=fusion_consensus_pos_mm,
                fusion_consensus_rot_deg=fusion_consensus_rot_deg,
                fusion_consensus_gripper_mm=fusion_consensus_gripper_mm,
                ensemble_consistency_pos_mm=ensemble_consistency_pos_mm,
                ensemble_consistency_rot_deg=ensemble_consistency_rot_deg,
                max_first_target_pos_mm=max_first_target_pos_mm,
                max_first_target_rot_deg=max_first_target_rot_deg,
                max_target_speed_m_s=max_target_speed_m_s,
                max_target_rot_speed_deg_s=max_target_rot_speed_deg_s,
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

            episode_idx = 0
            # Replay entries may be absent for a stopped/no-action trial. Keep
            # videos separate even when replay_buffer.n_episodes does not grow.
            import tempfile
            trial_video_root = env.video_dir / os.path.basename(
                tempfile.mkdtemp(prefix='trials_', dir=str(env.video_dir)))
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
                    print(f'Episode {episode_idx}')
                    print("\n" + "=" * 60)
                    print("Policy control started!")
                    print("Press 'S' to stop")
                    print("=" * 60)

                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)
                    action_ensembler = (
                        TimeAlignedActionEnsembler(
                            dt=dt,
                            time_origin=eval_t_start,
                            max_chunks=ensemble_steps,
                            consensus_translation_m=fusion_consensus_pos_mm / 1000.0,
                            consensus_rotation_rad=np.deg2rad(
                                fusion_consensus_rot_deg
                            ),
                            consensus_gripper_m=fusion_consensus_gripper_mm / 1000.0,
                        )
                        if temporal_agg else None
                    )
                    submitted_timeline = SubmittedActionTimeline(
                        dt=dt, time_origin=eval_t_start
                    )

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
                        measured_gripper_width_m = np.asarray(
                            obs['robot0_gripper_width'][-1], dtype=np.float32
                        ).reshape(-1)[:1]
                        policy_gripper_width_m = measured_gripper_width_m
                        if debug_actions and 'initial_position' in grippers_config[0]:
                            # Dry-run hardware intentionally does not move. Keep the
                            # policy input at its configured in-distribution opening.
                            policy_gripper_width_m = np.asarray(
                                [grippers_config[0]['initial_position']], dtype=np.float32
                            )
                        current_gripper = (
                            policy_gripper_width_m * hardware_to_model_gripper_scale
                        )
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
                                gripper_width_m=measured_gripper_width_m,
                                policy_gripper_width_m=policy_gripper_width_m,
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
                        decoded_action = np.concatenate([
                            absolute_action10[:, :3],
                            absolute_rotvec,
                            absolute_action10[:, 9:10]
                            * model_to_hardware_gripper_scale,
                        ], axis=-1)
                        action = decoded_action
                        original_action_horizon = len(decoded_action)
                        latency_trimmed_steps = 0

                        # env.exec_actions subtracts robot_action_latency. Keep
                        # another 50 ms so the local command arrives before its tick.
                        action_exec_latency = max(
                            float(rc.get('robot_action_latency', 0.0))
                            for rc in robots_config
                        ) + 0.05
                        if reanchor_action_chunks:
                            anchor_obs = env.get_obs()
                            measured_anchor_pose6 = np.concatenate([
                                anchor_obs['robot0_eef_pos'][-1],
                                anchor_obs['robot0_eef_rot_axis_angle'][-1],
                            ])
                            measured_anchor_timestamp = float(
                                anchor_obs['timestamp'][-1]
                            )
                            planning_time = time.time()
                            first_tick = int(np.floor(
                                (planning_time + action_exec_latency - eval_t_start) / dt
                            )) + 1
                            first_action_timestamp = eval_t_start + first_tick * dt
                            latency_trimmed_steps = compute_latency_trimmed_steps(
                                observation_timestamp=float(obs_timestamps[-1]),
                                planning_time=planning_time,
                                execution_margin=action_exec_latency,
                                dt=dt,
                                horizon=original_action_horizon,
                            )
                            execution_decoded_action = decoded_action[
                                latency_trimmed_steps:
                            ]
                            submitted_timeline.prune(planning_time - dt)
                            committed_anchor = submitted_timeline.get(
                                first_action_timestamp
                            )
                            if committed_anchor is None:
                                anchor_pose6 = measured_anchor_pose6
                                anchor_source = 'measured'
                            else:
                                anchor_pose6 = committed_anchor[:6]
                                anchor_source = 'submitted'
                            action = reanchor_action_chunk(
                                execution_decoded_action, anchor_pose6
                            )
                            action_horizon = len(action)
                            newest_timestamps = first_action_timestamp + (
                                np.arange(action_horizon, dtype=np.float64) * dt
                            )
                        else:
                            anchor_pose6 = current_pose6
                            measured_anchor_pose6 = current_pose6
                            measured_anchor_timestamp = float(obs_timestamps[-1])
                            anchor_source = 'observation'
                            committed_anchor = None
                            first_action_timestamp = None
                            action_horizon = len(action)
                            newest_timestamps = obs_timestamps[-1] + (
                                np.arange(1, action_horizon + 1, dtype=np.float64) * dt
                            )

                        if temporal_agg:
                            rejected_chunks = action_ensembler.reject_inconsistent_chunks(
                                measured_pose6=measured_anchor_pose6,
                                observation_timestamp=measured_anchor_timestamp,
                                max_translation_m=ensemble_consistency_pos_mm / 1000.0,
                                max_rotation_rad=np.deg2rad(ensemble_consistency_rot_deg),
                            )
                            if rejected_chunks:
                                worst_translation_m = max(item[0] for item in rejected_chunks)
                                worst_rotation_rad = max(item[1] for item in rejected_chunks)
                                print(
                                    f'[Ensemble] Rejected {len(rejected_chunks)} stale chunk(s): '
                                    f'worst tracking error={worst_translation_m * 1000:.1f}mm/'
                                    f'{np.rad2deg(worst_rotation_rad):.2f}deg'
                                )
                            if reanchor_action_chunks:
                                action_ensembler.add_retimed_chunk(
                                    actions=action,
                                    first_action_timestamp=first_action_timestamp,
                                )
                            else:
                                action_ensembler.add_chunk(
                                    actions=action,
                                    observation_timestamp=obs_timestamps[-1],
                                )
                        else:
                            rejected_chunks = []

                        # Every response contributes to temporal aggregation, but do not
                        # replace a trajectory before the controller has time to execute it.
                        if inference_idx >= steps_per_inference:
                            inference_idx = 0
                            curr_time = time.time()
                            schedulable_after = curr_time + action_exec_latency

                            is_new = newest_timestamps > schedulable_after
                            prediction_steps = original_action_horizon
                            eligible_action_indices = np.flatnonzero(is_new)
                            kept_indices = (
                                eligible_action_indices + latency_trimmed_steps
                            )
                            filtered_steps = prediction_steps - len(kept_indices)

                            if temporal_agg:
                                this_target_poses, action_timestamps, contributor_counts = (
                                    action_ensembler.get_fused_trajectory(
                                        after_time=schedulable_after,
                                        max_steps=exec_horizon_steps,
                                    )
                                )
                                candidate_counts = action_ensembler.last_candidate_counts.copy()
                                # Fused ticks can come from several chunks, so there is
                                # no single source-frame index for the selected targets.
                                selected_original_indices = None
                            else:
                                selected_indices = eligible_action_indices[
                                    :exec_horizon_steps
                                ]
                                selected_original_indices = (
                                    selected_indices + latency_trimmed_steps
                                )
                                this_target_poses = action[selected_indices]
                                action_timestamps = newest_timestamps[selected_indices]
                                contributor_counts = np.ones(
                                    len(this_target_poses), dtype=np.int64
                                )
                                candidate_counts = contributor_counts.copy()

                            used_fallback = len(this_target_poses) == 0
                            if used_fallback:
                                selected_original_indices = np.array([
                                    prediction_steps - 1
                                ])
                                next_step_idx = int(np.floor(
                                    (schedulable_after - eval_t_start) / dt
                                )) + 1
                                action_timestamp = eval_t_start + next_step_idx * dt
                                print(f'Over budget: {action_timestamp - curr_time:.3f}s')
                                action_timestamps = np.array([action_timestamp])
                                this_target_poses = action[[-1]].copy()
                                contributor_counts = np.ones(1, dtype=np.int64)
                                candidate_counts = contributor_counts.copy()

                            # Collision constraints operate on the fused targets, not
                            # on cached predictions, so the cache remains immutable.
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

                            horizon_dropped = max(
                                len(eligible_action_indices) - exec_horizon_steps, 0
                            )
                            contributor_summary = (
                                f"{int(contributor_counts.min())}-"
                                f"{int(contributor_counts.max())}"
                            )
                            candidate_summary = (
                                f"{int(candidate_counts.min())}-"
                                f"{int(candidate_counts.max())}"
                            )
                            print(
                                f"[Timing] step={iter_idx} predicted={prediction_steps} "
                                f"filtered={filtered_steps} eligible={len(kept_indices)} "
                                f"latency_trimmed={latency_trimmed_steps} "
                                f"horizon_dropped={horizon_dropped} "
                                f"newest_eligible_indices={kept_indices.tolist()} (0-based) "
                                f"fused_steps={len(this_target_poses)} "
                                f"contributors={contributor_summary}/"
                                f"candidates={candidate_summary} "
                                f"fallback={used_fallback}; "
                                f"obs_age={curr_time - obs_timestamps[-1]:.3f}s "
                                f"selection_margin={action_exec_latency:.3f}s dt={dt:.4f}s"
                            )
                            if len(this_target_poses):
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

                            latest = env.get_obs()
                            measured = np.concatenate([
                                latest['robot0_eef_pos'][-1],
                                latest['robot0_eef_rot_axis_angle'][-1],
                            ])
                            checked_at = time.time()
                            robot_latency = float(
                                robots_config[0].get('robot_action_latency', 0.0)
                            )
                            lead_ms = (
                                action_timestamps - robot_latency - checked_at
                            ) * 1000
                            translation_mm = np.linalg.norm(
                                this_target_poses[:, :3] - measured[:3], axis=1
                            ) * 1000
                            rotation_deg = np.rad2deg((
                                st.Rotation.from_rotvec(measured[3:6]).inv()
                                * st.Rotation.from_rotvec(this_target_poses[:, 3:6])
                            ).magnitude())
                            safety = check_trajectory_safety(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                measured_pose6=measured,
                                checked_at=checked_at,
                                robot_action_latency=robot_latency,
                                max_first_translation_m=max_first_target_pos_mm / 1000.0,
                                max_first_rotation_rad=np.deg2rad(max_first_target_rot_deg),
                                max_translation_speed_m_s=max_target_speed_m_s,
                                max_rotation_speed_rad_s=np.deg2rad(
                                    max_target_rot_speed_deg_s
                                ),
                            )
                            if not safety['passed']:
                                print('[Safety] TARGETS REJECTED: ' + '; '.join(safety['reasons']))
                                if temporal_agg:
                                    action_ensembler.clear()

                            if debug_actions:
                                # Observe only: never send a HOLD or model command here.
                                record = dict(
                                    episode=episode_idx, step=iter_idx, wall_time=checked_at,
                                    server_metadata=policy_client.metadata,
                                    configured_relative=action_is_relative,
                                    tx_flange_tcp=robots_config[0].get('tx_flange_tcp'),
                                    hardware_to_model_gripper_scale=hardware_to_model_gripper_scale,
                                    model_to_hardware_gripper_scale=model_to_hardware_gripper_scale,
                                    input_state10=current_state.tolist(),
                                    inference_base_pose6=current_pose6.tolist(),
                                    reanchor_enabled=reanchor_action_chunks,
                                    reanchor_pose6=anchor_pose6.tolist(),
                                    reanchor_source=anchor_source,
                                    measured_anchor_pose6=measured_anchor_pose6.tolist(),
                                    measured_anchor_timestamp=measured_anchor_timestamp,
                                    committed_anchor7=(
                                        None if committed_anchor is None
                                        else committed_anchor.tolist()
                                    ),
                                    retimed_first_action_timestamp=first_action_timestamp,
                                    image_shape=list(image.shape), image_dtype=str(image.dtype),
                                    raw_actions10=np.asarray(raw_action).tolist(),
                                    decoded_absolute_actions7=decoded_action.tolist(),
                                    latency_trimmed_steps=latency_trimmed_steps,
                                    eligible_original_action_indices=kept_indices.tolist(),
                                    selected_original_action_indices=(
                                        None if selected_original_indices is None
                                        else selected_original_indices.tolist()
                                    ),
                                    execution_actions7=action.tolist(),
                                    selected_targets7=this_target_poses.tolist(),
                                    measured_pose6=measured.tolist(),
                                    measured_gripper_m=np.asarray(latest['robot0_gripper_width'][-1]).tolist(),
                                    policy_gripper_m=policy_gripper_width_m.tolist(),
                                    observation_timestamp=float(obs_timestamps[-1]),
                                    latest_observation_timestamp=float(latest['timestamp'][-1]),
                                    network_latency_s=inference_time,
                                    action_timestamps=action_timestamps.tolist(),
                                    ensemble_contributor_counts=contributor_counts.tolist(),
                                    ensemble_candidate_counts=candidate_counts.tolist(),
                                    phase_rejected_predictions=(
                                        candidate_counts - contributor_counts
                                    ).tolist(),
                                    robot_action_latency_s=robot_latency,
                                    compensated_lead_ms=lead_ms.tolist(),
                                    translation_delta_mm=translation_mm.tolist(),
                                    rotation_delta_deg=rotation_deg.tolist(),
                                    rejected_ensemble_chunks=len(rejected_chunks),
                                    rejected_ensemble_chunk_errors=[
                                        {
                                            'translation_mm': error[0] * 1000.0,
                                            'rotation_deg': np.rad2deg(error[1]),
                                        }
                                        for error in rejected_chunks
                                    ],
                                    safety=safety,
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
                            elif safety['passed']:
                                # Track the entire submitted chunk, including a
                                # possible partial submission if enqueue fails.
                                last_action_end_time = max(
                                    last_action_end_time, float(np.max(action_timestamps)))
                                env.exec_actions(
                                    actions=this_target_poses,
                                    timestamps=action_timestamps,
                                    compensate_latency=True
                                )
                                submitted_timeline.update(
                                    actions=this_target_poses,
                                    timestamps=action_timestamps,
                                )
                                print(
                                    f"Step {iter_idx}: Submitted {len(this_target_poses)} actions, "
                                    f"gripper={this_target_poses[0, 6]:.4f}, "
                                    f"anchor={anchor_source}, "
                                    f"contributors={int(contributor_counts.min())}-"
                                    f"{int(contributor_counts.max())}"
                                )
                            else:
                                print(
                                    f'Step {iter_idx}: Submitted 0 actions; safety guard '
                                    'cleared the temporal ensemble.'
                                )
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

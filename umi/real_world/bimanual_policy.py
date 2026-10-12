""" PC-side OpenPI serialization and per-arm geometry; no hardware imports. """
import os

import numpy as np
from scipy.spatial.transform import Rotation


def validate_hardware_config(robots, grippers):
    """Check only arm counts and independent hardware routing."""
    if len(robots) not in (1, 2) or len(grippers) != len(robots):
        raise ValueError('Configure one or two robots and exactly one gripper per robot')
    endpoints = [(rc['robot_ip'], int(rc.get('robot_port', 4242))) for rc in robots]
    if any(not ip for ip, _ in endpoints):
        raise ValueError('Each robot needs a robot_ip')
    if len(set(endpoints)) != len(endpoints):
        raise ValueError('duplicate robot endpoint: use distinct IP/port pairs')
    custom = [gc for gc in grippers if gc.get('gripper_type') == 'custom']
    if any(gc.get('mode', 'single') != 'single' for gc in custom):
        raise ValueError('Use one single-mode custom gripper entry per arm')
    ports = [os.path.realpath(os.path.expanduser(gc['port1'])) for gc in custom]
    if len(set(ports)) != len(ports):
        raise ValueError('Custom grippers require separate serial ports')


def policy_options(config, num_robots):
    """Choose the wire keys from robot count; every arm block follows list order."""
    if num_robots not in (1, 2):
        raise ValueError('Configure one or two robots')
    default_image_keys = (['observation/image_right'] if num_robots == 1 else
                          ['observation/image_right', 'observation/image_left'])
    if 'action_is_relative' in config and not isinstance(config['action_is_relative'], bool):
        raise ValueError('action_is_relative must be a YAML boolean, not a quoted string')
    
    result = dict(
        image_keys=config.get('image_keys', default_image_keys),
        state_key=config.get('state_key', 'current_tcp_pose9_gripper'),
    )
    keys = result['image_keys']
    if (not isinstance(keys, list) or len(keys) != num_robots
            or any(not isinstance(k, str) or not k for k in keys)
            or len(set(keys)) != num_robots):
        raise ValueError('policy.image_keys needs one distinct request key per arm, in robot list order')
    if not isinstance(result['state_key'], str) or not result['state_key']:
        raise ValueError('state_key must be a nonempty string')
    return result


def gripper_scales(value, count):
    scales = np.broadcast_to(np.asarray(value, dtype=float), (count,)).copy()
    if not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError('Gripper scales must be positive; use a scalar or one value per robot')
    return scales


def pack_state(arm_states):
    states = np.asarray(arm_states)
    return states.reshape(-1).astype(np.float32)


def unpack_actions(actions, count):
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 10 * count or len(actions) == 0:
        raise ValueError(f'Expected nonempty [T, {10 * count}] actions, got {actions.shape}')
    
    if not np.isfinite(actions).all():
        raise ValueError('Actions contain non-finite values')
    return actions.reshape(-1, count, 10)


def rotation6d_to_matrix(vectors):
    vectors = np.asarray(vectors, dtype=float)
    x, y = vectors[..., :3], vectors[..., 3:]
    xn = np.linalg.norm(x, axis=-1, keepdims=True)
    if (xn < 1e-8).any():
        raise ValueError('Invalid 6D rotation: zero first column')
    x = x / xn
    z = np.cross(x, y)
    zn = np.linalg.norm(z, axis=-1, keepdims=True)
    if (zn < 1e-8).any():
        raise ValueError('Invalid 6D rotation: collinear columns')
    z = z / zn
    return np.stack([x, np.cross(z, x), z], axis=-1)


def build_observation(obs, count, resolution, scales, options):
    import cv2
    images, poses, widths, states = [], [], [], []
    for i in range(count):
        images.append(cv2.resize(obs[f'camera{i}_rgb'][-1], (resolution, resolution),
                                 interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False))
        pose = np.concatenate([obs[f'robot{i}_eef_pos'][-1], obs[f'robot{i}_eef_rot_axis_angle'][-1]])
        width = float(np.asarray(obs[f'robot{i}_gripper_width'][-1]).reshape(-1)[0])
        rot6 = Rotation.from_rotvec(pose[3:]).as_matrix()[:, :2].T.reshape(6)
        states.append(np.r_[pose[:3], rot6, width * scales[i]])
        poses.append(pose)
        widths.append(width)
    if not np.isfinite(states).all():
        raise ValueError('Non-finite robot observation')
    return images, pack_state(states), np.asarray(poses), np.asarray(widths)


def decode_actions(raw, poses, scales, relative, options):
    """Each relative pose is anchored to that arm's request TCP, never the other arm."""
    count = len(poses)
    blocks = unpack_actions(raw, count)
    output = np.empty((len(blocks), count, 7))
    for i in range(count):
        positions = blocks[:, i, :3]
        rotations = rotation6d_to_matrix(blocks[:, i, 3:9])
        if relative:
            base_rotation = Rotation.from_rotvec(poses[i, 3:]).as_matrix()
            positions = positions @ base_rotation.T + poses[i, :3]
            rotations = base_rotation @ rotations
        output[:, i, :3] = positions
        output[:, i, 3:6] = Rotation.from_matrix(rotations).as_rotvec()
        # Gripper targets remain absolute, even when TCP targets are relative.
        output[:, i, 6] = blocks[:, i, 9] * scales[i]
    return output.reshape(len(output), 7 * count)


def average_actions(actions, weights):
    result = np.average(actions, axis=0, weights=weights)
    for start in range(0, actions.shape[1], 7):
        result[start + 3:start + 6] = Rotation.from_rotvec(actions[:, start + 3:start + 6]).mean(weights=weights).as_rotvec()
    return result[None]


class PolicyClient:
    def __init__(self, server_url, prompt, timeout=60.0, num_robots=1, policy_config=None):
        from websockets.sync.client import connect
        from umi.common import openpi_msgpack_numpy
        self.options = policy_options(policy_config or {}, num_robots)
        self.num_robots, self.prompt, self.timeout = num_robots, prompt, timeout
        if not server_url or server_url.startswith(('http://', 'https://')):
            raise ValueError('Set a ws:// or wss:// OpenPI server URL')
        if not server_url.startswith(('ws://', 'wss://')):
            server_url = 'ws://' + server_url
        self.codec = openpi_msgpack_numpy
        self.packer = self.codec.Packer()
        self.ws = connect(server_url.rstrip('/'), open_timeout=timeout,
                          close_timeout=min(timeout, 5), compression=None, max_size=None)
        try:
            message = self.ws.recv(timeout=timeout)
            self.metadata = self.codec.unpackb(message)
            keys = list(self.metadata) if isinstance(self.metadata, dict) else type(self.metadata).__name__
        except BaseException:
            self.close()
            raise
        print(f'OpenPI connected; {num_robots} arm(s); metadata keys: {keys}')

    def close(self):
        if self.ws is not None:
            self.ws.close()
            self.ws = None

    def build_payload(self, images, state):
        if isinstance(images, np.ndarray) and images.ndim == 3:
            images = [images]
        if len(images) != self.num_robots:
            raise ValueError('One image per arm is required')
        
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (10 * self.num_robots,) or not np.isfinite(state).all():
            raise ValueError('Incorrect state shape or non-finite state')
        
        payload = {self.options['state_key']: state, 'prompt': self.prompt}
        for robot_idx, key in enumerate(self.options['image_keys']):
            image = np.asarray(images[robot_idx])
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError('Images must be HWC uint8 RGB')
            payload[key] = image
        return payload

    def call_server(self, images, state):
        # Fail visibly on contract/connection errors; never silently send one arm's output to both.
        self.ws.send(self.packer.pack(self.build_payload(images, state)))
        message = self.ws.recv(timeout=self.timeout)
        if isinstance(message, str):
            raise RuntimeError(f'OpenPI server error: {message}')
        result = self.codec.unpackb(message)

        # ==============================================================================================
        # action format [pose_R, gripper_R, pose_L, gripper_L]
        raw_action = result.get('actions', result.get('action')) if isinstance(result, dict) else result
        # ==============================================================================================

        action = np.asarray(raw_action, dtype=np.float32)
        if action.ndim == 1:
            action = action[None]
        unpack_actions(action, self.num_robots)
        return action

"""Right-first one/two-arm paired history for compatible OpenPI proprio servers."""
import numpy as np
from umi.real_world.bimanual_policy import PolicyClient, build_observation


def image_keys(count):
    if count == 1:
        return ['observation/image']
    if count == 2:
        return ['observation/image_right', 'observation/image_left']
    raise ValueError('Expected one or two arms')


def paired_history(obs, resolution, scales, options, count=1):
    image_keys(count)
    keys = [key for i in range(count) for key in (
        f'camera{i}_rgb', f'robot{i}_eef_pos', f'robot{i}_eef_rot_axis_angle', f'robot{i}_gripper_width')]
    if any(len(obs[key]) != 2 for key in keys):
        raise ValueError('Proprio requires two aligned samples for every camera, TCP and gripper')
    previous = dict(obs)
    for key in keys:
        previous[key] = obs[key][:1]
    old_images, old_state, _, _ = build_observation(previous, count, resolution, scales, options)
    _, current_state, _, _ = build_observation(obs, count, resolution, scales, options)
    return old_images, np.stack([old_state, current_state])


class ProprioPolicyClient(PolicyClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.validate_contract()
        except BaseException:
            self.close()
            raise
        self.reset_episode()

    def validate_contract(self):
        keys = image_keys(self.num_robots)
        if self.options['image_keys'] != keys or self.options['state_key'] != 'current_tcp_pose9_gripper':
            raise ValueError('Use canonical proprio image/state keys; remove custom image_keys/state_key')
        meta = self.metadata
        if (not isinstance(meta, dict) or meta.get('proprioception') is not True
                or meta.get('image_history_horizon') != 2 or meta.get('is_relative') is not False):
            raise ValueError('Server must advertise proprioception, image_history_horizon=2, is_relative=false')
        if meta.get('state_dim') != 32 * self.num_robots:
            raise ValueError('Server proprio state dimension does not match configured arm count')
        if self.num_robots == 2 and (meta.get('num_robots') != 2 or meta.get('arm_order') != ['right', 'left']):
            raise ValueError('Dual proprio server must advertise num_robots=2 and right-first arm_order')

    def reset_episode(self):
        self.start_state = None
        self.start_images = None
        self.start_time = None
        self.history = None
        self.previous_images = None

    def set_history(self, previous_images, history, previous_time, current_time):
        history = np.asarray(history, dtype=np.float32)
        if history.shape != (2, 10 * self.num_robots) or not np.isfinite(history).all():
            raise ValueError(f'Expected finite history [2,{10 * self.num_robots}]')
        if isinstance(previous_images, np.ndarray) and previous_images.ndim == 3:
            previous_images = [previous_images]
        if len(previous_images) != self.num_robots:
            raise ValueError('One historical image per arm is required')
        if not np.isfinite([previous_time, current_time]).all() or previous_time > current_time:
            raise ValueError('Invalid history timestamps')
        self.history = history.copy()
        self.previous_images = [np.asarray(image).copy() for image in previous_images]
        self.previous_time, self.current_time = previous_time, current_time

    def build_payload(self, images, state):
        payload = super().build_payload(images, state)
        if self.history is None or not np.allclose(self.history[-1], state, atol=1e-5, rtol=0):
            raise ValueError('History must be set and end at the request TCP state')
        keys = image_keys(self.num_robots)
        current = [payload[key] for key in keys]
        previous = self.previous_images
        for old, new in zip(previous, current):
            if old.shape != new.shape or old.dtype != np.uint8:
                raise ValueError('Historical image must match current HWC uint8 image format')
        if self.start_state is None:
            self.start_state = np.asarray(state, dtype=np.float32).copy()
            self.start_images = [image.copy() for image in current]
            self.start_time = self.current_time
        history = self.history.copy()
        if self.previous_time < self.start_time:
            previous = self.start_images
            history[0] = self.start_state
        payload.update({key + '_previous': old for key, old in zip(keys, previous)})
        payload.update({
            'tcp_pose9_gripper_history': history,
            'episode_start_tcp_pose9_gripper': self.start_state.copy(),
        })
        return payload

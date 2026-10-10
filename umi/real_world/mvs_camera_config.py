"""Validation and pickle-safe image transforms for direct MVS cameras."""
from dataclasses import dataclass
import math

import cv2
import numpy as np


def normalize_mvs_configs(configs, camera_reorder=None):
    result = []
    for item in configs:
        cfg = dict(item)
        if cfg.get('backend', 'mvs').lower() != 'mvs':
            raise ValueError('Only support MVS cameras')
        if not isinstance(cfg.get('serial'), str) or not cfg['serial'].strip():
            raise ValueError('Each MVS camera requires an exact serial string')
        
        for key, default in [('width', 480), ('height', 480), ('fps', 30),
                             ('frame_timeout_ms', 1000)]:
            value = cfg.get(key, default)
            # if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0 or int(value) != value:
            #     raise ValueError(f'{key} must be a positive integer')
            cfg[key] = int(value)

        if cfg['width'] % 2 or cfg['height'] % 2:
            raise ValueError('MVS output dimensions must be even for yuv420p video recording')
        
        latency = float(cfg.get('receive_latency', 0.0))
        cfg['receive_latency'] = latency
        timeout = float(cfg.get('launch_timeout', 15))
        cfg['launch_timeout'] = timeout
        roi = cfg.get('sensor_roi', {})
        if roi.get('enabled', False):
            for key in ['width', 'height', 'offset_x', 'offset_y']:
                value = roi.get(key, 0)
                # minimum = 1 if key in ['width', 'height'] else 0
                # if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                #     raise ValueError(f'sensor_roi.{key} must be an integer >= {minimum}')
        result.append(cfg)

    if len({cfg['serial'] for cfg in result}) != len(result):
        raise ValueError('MVS serial numbers must be unique')
    
    if camera_reorder is not None:
        if (not camera_reorder or len(set(camera_reorder)) != len(camera_reorder)
                or any(i < 0 or i >= len(result) for i in camera_reorder)):
            raise ValueError('camera_reorder must contain unique valid camera indices')
        result = [result[i] for i in camera_reorder]
    return result


@dataclass
class MvsImageTransform:
    resolution: tuple
    rgb: bool = True
    float32: bool = False

    def __call__(self, data):
        # Driver already applies software ROI/padding and mounting rotation. Use the
        # square OmniUmiX image, without official GoPro/mirror/gripper masks.
        image = cv2.resize(data['color'], tuple(self.resolution), interpolation=cv2.INTER_LINEAR)
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.float32:
            image = image.astype(np.float32) / 255.0
        data['color'] = np.ascontiguousarray(image)
        return data

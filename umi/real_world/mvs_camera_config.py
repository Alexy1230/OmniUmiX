"""Validation and pickle-safe image transforms for direct MVS cameras."""
from dataclasses import dataclass
import math

import av
import cv2
import numpy as np


_RESIZE_INTERPOLATIONS = {
    'nearest': cv2.INTER_NEAREST,
    'linear': cv2.INTER_LINEAR,
    'area': cv2.INTER_AREA,
    'cubic': cv2.INTER_CUBIC,
    'lanczos': cv2.INTER_LANCZOS4,
}


def resize_interpolation(value):
    """Validate a named resize method and return its OpenCV equivalent."""
    name = str(value).lower()
    if name not in _RESIZE_INTERPOLATIONS:
        raise ValueError(
            'resize_interpolation must be one of: '
            + ', '.join(sorted(_RESIZE_INTERPOLATIONS))
        )
    return _RESIZE_INTERPOLATIONS[name]


def resize_image(image, resolution, interpolation='linear'):
    """Resize a BGR uint8 camera frame with the selected implementation.

    The training video pipeline uses FFmpeg ``scale=...:flags=lanczos``.  PyAV's
    LANCZOS reformatter calls the same libswscale implementation; OpenCV's
    INTER_LANCZOS4 is a different filter and noticeably amplifies high-frequency
    image noise.
    """
    resolution = tuple(resolution)
    if image.shape[1::-1] == resolution:
        return image

    name = str(interpolation).lower()
    resize_interpolation(name)
    if name == 'lanczos':
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('libswscale Lanczos expects a uint8 HWC BGR image')
        frame = av.VideoFrame.from_ndarray(
            np.ascontiguousarray(image), format='bgr24')
        return frame.reformat(
            width=resolution[0],
            height=resolution[1],
            format='bgr24',
            interpolation=av.video.reformatter.Interpolation.LANCZOS,
        ).to_ndarray(format='bgr24')

    return cv2.resize(
        image,
        resolution,
        interpolation=resize_interpolation(name),
    )


def normalize_mvs_configs(configs, camera_reorder=None):
    result = []
    for item in configs:
        cfg = dict(item)
        if cfg.get('backend', 'mvs').lower() != 'mvs':
            raise ValueError('Only support MVS cameras')
        if not isinstance(cfg.get('serial'), str) or not cfg['serial'].strip():
            raise ValueError('Each MVS camera requires an exact serial string')
        
        for key, default in [('source_width', 1440), ('source_height', 1080),
                             ('width', 480), ('height', 480), ('fps', 30),
                             ('frame_timeout_ms', 1000)]:
            value = cfg.get(key, default)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value <= 0 or int(value) != value):
                raise ValueError(f'{key} must be a positive integer')
            cfg[key] = int(value)

        cfg['resize_interpolation'] = str(
            cfg.get('resize_interpolation', 'linear')
        ).lower()
        resize_interpolation(cfg['resize_interpolation'])

        if cfg['width'] % 2 or cfg['height'] % 2:
            raise ValueError('MVS output dimensions must be even for yuv420p video recording')
        
        latency = float(cfg.get('receive_latency', 0.0))
        if not math.isfinite(latency) or latency < 0:
            raise ValueError('receive_latency must be finite and non-negative')
        cfg['receive_latency'] = latency
        timeout = float(cfg.get('launch_timeout', 15))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError('launch_timeout must be finite and positive')
        cfg['launch_timeout'] = timeout
        roi = cfg.get('sensor_roi', {})
        if roi.get('enabled', False):
            for key in ('width', 'height'):
                value = roi.get(key)
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f'sensor_roi.{key} must be a positive integer')
            for key in ('offset_x', 'offset_y'):
                value = roi.get(key)
                if (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value)):
                    raise ValueError(f'sensor_roi.{key} must be finite')
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
    interpolation: str = 'linear'

    def __call__(self, data):
        # Driver already applies software ROI/padding and mounting rotation. Use the
        # square OmniUmiX image, without official GoPro/mirror/gripper masks.
        image = data['color']
        if image.shape[1::-1] != tuple(self.resolution):
            image = resize_image(
                image, tuple(self.resolution), self.interpolation)
        if self.rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.float32:
            image = image.astype(np.float32) / 255.0
        data['color'] = np.ascontiguousarray(image)
        return data


def camera_configs_from_yaml(config):
    """Collection-style per-hand cameras, ordered right then left."""
    if 'mvs' not in config:
        return config.get('cameras')
    if 'cameras' in config:
        raise ValueError('Specify mvs or cameras, not both')
    cameras = config['mvs']
    if not isinstance(cameras, dict) or not cameras or set(cameras) - {'right_hand', 'left_hand'}:
        raise ValueError('mvs must contain right_hand and optionally left_hand')
    if 'right_hand' not in cameras:
        raise ValueError('Single arm uses mvs.right_hand')
    result = []
    for name in ('right_hand', 'left_hand'):
        if name not in cameras:
            continue
        item = dict(cameras[name])
        item.setdefault('exposure_auto', True)
        item.setdefault('target_brightness', 40)
        item.setdefault('gain_auto', 'continuous')
        item.setdefault('exposure_time', 12500)
        item.setdefault('rotation', 180)
        if not isinstance(item['exposure_auto'], bool):
            raise ValueError('exposure_auto must be a YAML boolean')
        if item['gain_auto'] not in ('off', 'once', 'continuous'):
            raise ValueError('gain_auto must be off, once or continuous (quote off in YAML)')
        if item['rotation'] not in (0, 90, 180, 270):
            raise ValueError('rotation must be 0, 90, 180 or 270')
        result.append(item)
    return result

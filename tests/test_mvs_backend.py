"""Hardware-free MVS contract tests; never import or open the real SDK."""
import ctypes as ct
from types import SimpleNamespace
from pathlib import Path
import time
import multiprocessing as mp
import threading
import sys
import importlib
from multiprocessing.managers import SharedMemoryManager

import av
import numpy as np
import pytest

from umi.real_world.mvs_capture import MvsCapture, MvsFrameTimeout, crop_with_black_padding
from umi.real_world.mvs_camera_config import normalize_mvs_configs, MvsImageTransform, resize_image
from umi.real_world.mvs_camera import MvsCamera, MultiMvsCamera


class UsbInfo(ct.Structure):
    _fields_ = [('chSerialNumber', ct.c_ubyte * 64)]


class SpecialInfo(ct.Structure):
    _fields_ = [('stUsb3VInfo', UsbInfo)]


class DeviceInfo(ct.Structure):
    _fields_ = [('SpecialInfo', SpecialInfo)]


def fake_sdk(fail=None):
    calls = []
    device = DeviceInfo()
    device.SpecialInfo.stUsb3VInfo.chSerialNumber[:4] = b'TEST'
    image = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)

    class Camera:
        @staticmethod
        def MV_CC_Initialize():
            calls.append('initialize')
            return 0

        @staticmethod
        def MV_CC_Finalize():
            calls.append('finalize')
            return 0

        @staticmethod
        def MV_CC_EnumDevices(kind, devices):
            calls.append('enumerate')
            devices.nDeviceNum = 1
            devices.pDeviceInfo = [ct.pointer(device)]
            return 0

        def MV_CC_GetImageBuffer(self, frame, timeout):
            calls.append('read')
            frame.stFrameInfo = SimpleNamespace(nWidth=4, nHeight=4, nFrameLen=48, enPixelType=1)
            frame.pBufAddr = 0
            return 0

        def MV_CC_ConvertPixelTypeEx(self, params):
            calls.append('convert')
            if fail == 'convert':
                return 7
            ct.memmove(params.pDstBuffer, image.ctypes.data, image.nbytes)
            return 0

        def __getattr__(self, name):
            def invoke(*args):
                calls.append((name, args))
                return 7 if name == fail else 0
            return invoke

    sdk = SimpleNamespace(MvCamera=Camera, MV_CC_DEVICE_INFO=DeviceInfo,
                          MV_CC_DEVICE_INFO_LIST=SimpleNamespace, MV_FRAME_OUT=SimpleNamespace,
                          MV_CC_PIXEL_CONVERT_PARAM_EX=SimpleNamespace)
    for name in ['MV_USB_DEVICE', 'MV_ACCESS_Exclusive', 'MV_TRIGGER_MODE_OFF',
                 'MV_EXPOSURE_AUTO_MODE_OFF', 'MV_GAIN_MODE_CONTINUOUS', 'MV_GAIN_MODE_OFF',
                 'MV_BALANCEWHITE_AUTO_CONTINUOUS', 'MV_BALANCEWHITE_AUTO_OFF', 'PixelType_Gvsp_BGR8_Packed']:
        setattr(sdk, name, 1)
    return sdk, calls, image


def config(**overrides):
    return normalize_mvs_configs([
        dict(serial='TEST', source_width=4, source_height=4, width=4, height=4, fps=30, **overrides)
    ])[0]


def test_sdk_selection_pixels_rotation_and_release():
    sdk, calls, source = fake_sdk()
    cap = MvsCapture(config(), sdk=sdk).open()
    before = time.time()
    image, stamp = cap.read()
    assert before <= stamp <= time.time()
    np.testing.assert_array_equal(image, source[::-1, ::-1])
    assert any(call[0] == 'MV_CC_FreeImageBuffer' for call in calls if isinstance(call, tuple))
    cap.close()
    assert calls[-1] == 'finalize'
    assert [call[0] for call in calls if isinstance(call, tuple)][-3:] == [
        'MV_CC_StopGrabbing', 'MV_CC_CloseDevice', 'MV_CC_DestroyHandle']


def test_mvs_camera_retries_one_sdk_frame_timeout():
    camera = object.__new__(MvsCamera)
    camera.mvs_config = {'frame_timeout_ms': 1}
    camera.dev_video_path = 'TEST'
    camera.stop_event = threading.Event()
    image = np.full((2, 2, 3), 7, dtype=np.uint8)

    class Capture:
        calls = 0

        def read(self):
            self.calls += 1
            if self.calls == 1:
                raise MvsFrameTimeout('temporary timeout')
            return image, 123.0

    frame = np.empty_like(image)
    result, received, captured = camera._read_capture(Capture(), frame)
    assert result is frame
    np.testing.assert_array_equal(frame, image)
    assert received == captured == 123.0


@pytest.mark.parametrize('failure', ['convert', 'MV_CC_OpenDevice', 'missing_serial'])
def test_sdk_failure_cleanup(failure):
    sdk, calls, _ = fake_sdk(fail=failure)
    cfg = config()
    if failure == 'missing_serial':
        cfg['serial'] = 'MISSING'
    cap = MvsCapture(cfg, sdk=sdk)
    with pytest.raises(RuntimeError):
        try:
            cap.open()
            cap.read()
        finally:
            cap.close()
    assert calls[-1] == 'finalize'
    if failure == 'convert':
        assert any(call[0] == 'MV_CC_FreeImageBuffer' for call in calls if isinstance(call, tuple))


def test_cleanup_continues_after_sdk_error():
    sdk, calls, _ = fake_sdk(fail='MV_CC_StopGrabbing')
    cap = MvsCapture(config(), sdk=sdk).open()
    with pytest.warns(RuntimeWarning):
        cap.close()
    assert calls[-1] == 'finalize'


def test_rgb_float_and_bgr_preview():
    bgr = np.full((4, 4, 3), [10, 20, 230], dtype=np.uint8)
    rgb = MvsImageTransform((2, 2), float32=True)({'color': bgr})['color']
    assert rgb.shape == (2, 2, 3) and rgb.dtype == np.float32
    np.testing.assert_allclose(rgb[0, 0], np.array([230, 20, 10])/255, atol=1e-7)
    preview = MvsImageTransform((2, 2), rgb=False)({'color': bgr})['color']
    np.testing.assert_array_equal(preview[0, 0], bgr[0, 0])


def test_named_resize_interpolation_and_same_size_passthrough():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    transformed = MvsImageTransform((4, 4), interpolation='lanczos')({'color': image.copy()})['color']
    np.testing.assert_array_equal(transformed, image[..., ::-1])
    with pytest.raises(ValueError, match='resize_interpolation'):
        normalize_mvs_configs([dict(serial='TEST', resize_interpolation='unknown')])


def test_lanczos_resize_uses_ffmpeg_libswscale():
    import av

    image = np.arange(12 * 16 * 3, dtype=np.uint8).reshape(12, 16, 3)
    expected = av.VideoFrame.from_ndarray(image, format='bgr24').reformat(
        width=7,
        height=5,
        format='bgr24',
        interpolation=av.video.reformatter.Interpolation.LANCZOS,
    ).to_ndarray(format='bgr24')

    actual = resize_image(image, (7, 5), 'lanczos')

    np.testing.assert_array_equal(actual, expected)


def test_center_square_crop_geometry():
    source = np.arange(1080 * 1440, dtype=np.int32).reshape(1080, 1440)
    cropped = crop_with_black_padding(source, {
        'enabled': True, 'offset_x': 720, 'offset_y': 540, 'width': 1080, 'height': 1080,
    })
    np.testing.assert_array_equal(cropped, source[:, 180:1260])


@pytest.mark.parametrize(
    ('profile_name', 'crop_enabled'),
    [
        ('eval_pi05_adjust_tube_fov167.yaml', False),
        ('eval_pi05_pick_and_place_red_cup_fov167.yaml', True),
    ],
)
def test_fov167_profiles_match_training_contract(profile_name, crop_enabled):
    import yaml

    profile = Path(__file__).resolve().parents[1] / 'example' / profile_name
    config = yaml.safe_load(profile.read_text())
    camera = normalize_mvs_configs([config['mvs']['right_hand']])[0]
    assert (camera['width'], camera['height']) == (240, 240)
    assert (camera['source_width'], camera['source_height']) == (1440, 1080)
    assert camera['resize_interpolation'] == 'lanczos'
    assert camera['sensor_roi']['enabled'] is crop_enabled
    if crop_enabled:
        assert (camera['sensor_roi']['offset_x'], camera['sensor_roi']['offset_y']) == (720, 540)
        assert (camera['sensor_roi']['width'], camera['sensor_roi']['height']) == (1080, 1080)
    assert config['grippers'][0]['width_to_rad'] == {
        'min_abs_width_m': 0.04385771672409207,
        'max_abs_width_m': 0.14959563328076972,
        'min_rad': 0.0,
        'max_rad': 3.5,
    }


def test_all_center_crop_profiles_use_training_contract():
    import yaml

    profiles = sorted((Path(__file__).resolve().parents[1] / 'example').glob('*.yaml'))
    checked = 0
    for profile in profiles:
        config = yaml.safe_load(profile.read_text())
        if not isinstance(config, dict) or ('mvs' not in config and 'cameras' not in config):
            continue
        raw_cameras = (
            list(config['mvs'].values())
            if 'mvs' in config else list(config['cameras'])
        )
        for camera in normalize_mvs_configs(raw_cameras):
            roi = camera.get('sensor_roi', {})
            if not roi.get('enabled', False):
                continue
            checked += 1
            assert (camera['source_width'], camera['source_height']) == (1440, 1080), profile
            assert (camera['width'], camera['height']) == (240, 240), profile
            assert camera['resize_interpolation'] == 'lanczos', profile
            assert (roi['offset_x'], roi['offset_y']) == (720, 540), profile
            assert (roi['width'], roi['height']) == (1080, 1080), profile
    assert checked > 0


@pytest.mark.parametrize('override', [{'fps': 0}, {'width': 3}, {'receive_latency': -1},
                                      {'launch_timeout': 0}, {'sensor_roi': {'enabled': True}}])
def test_invalid_config(override):
    values = dict(serial='TEST', width=4, height=4, fps=30)
    values.update(override)
    with pytest.raises(ValueError):
        normalize_mvs_configs([values])


def test_serial_order_and_duplicates():
    configs = [{'serial': 'A'}, {'serial': 'B'}]
    assert [c['serial'] for c in normalize_mvs_configs(configs, [1, 0])] == ['B', 'A']
    with pytest.raises(ValueError):
        normalize_mvs_configs(configs, [0, 0])
    with pytest.raises(ValueError):
        normalize_mvs_configs([{'serial': 'A'}, {'serial': 'A'}])


class SyntheticCapture:
    def read(self):
        time.sleep(1 / 30)
        return np.full((4, 4, 3), [10, 20, 230], dtype=np.uint8), time.time()

    def close(self):
        pass


def open_synthetic(self):
    return SyntheticCapture()


def open_broken(self):
    raise RuntimeError('synthetic SDK open failure')


class LockedTestCounter:
    """macOS test substitute for a Linux dependency's incompatible wheel.

    Real shared image buffers and recorder processes are still exercised.
    Production code and Linux tests continue to use the original atomics.
    """
    def __init__(self, manager):
        self.value = mp.Value('Q', 0)

    def load(self):
        return self.value.value

    def store(self, value):
        self.value.value = value

    def add(self, value):
        with self.value.get_lock():
            self.value.value += value


@pytest.fixture
def fork_workers(monkeypatch):
    if 'fork' not in mp.get_all_start_methods():
        pytest.skip('Integration test requires fork, matching the Linux deployment')
    previous = mp.get_start_method(allow_none=True)
    mp.set_start_method('fork', force=True)
    if sys.platform == 'darwin':
        for name in ['umi.shared_memory.shared_memory_ring_buffer',
                     'umi.shared_memory.shared_memory_queue',
                     'umi.compat.dp.shared_memory.shared_memory_queue']:
            monkeypatch.setattr(importlib.import_module(name), 'SharedAtomicCounter', LockedTestCounter)
    yield
    mp.set_start_method(previous, force=True)


def test_worker_history_visualization_recording_and_shutdown(monkeypatch, tmp_path, fork_workers):
    monkeypatch.setattr(MvsCamera, '_open_capture', open_synthetic)
    cfg = config(receive_latency=0.025, launch_timeout=5)
    with SharedMemoryManager() as manager:
        camera = MultiMvsCamera([cfg], manager, [MvsImageTransform((4, 4), float32=True)],
                                [MvsImageTransform((4, 4), rgb=False)], 5)
        with camera:
            assert camera.is_ready
            time.sleep(0.2)
            history = camera.get(k=3)[0]
            assert history['color'].shape == (3, 4, 4, 3)
            assert history['timestamp'].shape == (3,)
            assert np.all(np.diff(history['timestamp']) > 0)
            np.testing.assert_allclose(history['camera_receive_timestamp'] - history['timestamp'], .025, atol=1e-6)
            vis = camera.get_vis()
            assert vis['color'].shape == (1, 4, 4, 3)
            np.testing.assert_array_equal(camera.get_vis(out=vis)['color'][0, 0, 0], [10, 20, 230])
            for index in range(2):
                start = time.time()
                camera.restart_put(start)
                camera.start_recording([str(tmp_path / f'{index}.mp4')], start)
                time.sleep(0.5)
                camera.stop_recording()
                time.sleep(0.15)
        assert not camera.is_ready
        for worker in camera.cameras.values():
            assert not worker.is_alive() and not worker.video_recorder.is_alive()
    for index in range(2):
        with av.open(str(tmp_path / f'{index}.mp4')) as video:
            frames = list(video.decode(video=0))
            assert len(frames) >= 3
            rgb = frames[0].to_ndarray(format='rgb24')
            assert rgb[..., 0].mean() > rgb[..., 2].mean() + 150


def test_startup_error_propagates_and_stops_workers(monkeypatch, fork_workers):
    monkeypatch.setattr(MvsCamera, '_open_capture', open_broken)
    with SharedMemoryManager() as manager:
        camera = MultiMvsCamera([config(launch_timeout=2)], manager,
                                [MvsImageTransform((4, 4))], [MvsImageTransform((4, 4))], 5)
        start = time.monotonic()
        with pytest.raises(RuntimeError):
            camera.start()
        assert time.monotonic() - start < 4
        for worker in camera.cameras.values():
            assert not worker.is_alive() and not worker.video_recorder.is_alive()


def open_slow(self):
    time.sleep(5)
    return SyntheticCapture()


def test_startup_timeout_cleans_up(monkeypatch, fork_workers):
    monkeypatch.setattr(MvsCamera, '_open_capture', open_slow)
    with SharedMemoryManager() as manager:
        camera = MultiMvsCamera([config(launch_timeout=0.2)], manager,
                                [MvsImageTransform((4, 4))], [MvsImageTransform((4, 4))], 5)
        start = time.monotonic()
        with pytest.raises(TimeoutError):
            camera.start()
        assert time.monotonic() - start < 3
        for worker in camera.cameras.values():
            assert not worker.is_alive() and not worker.video_recorder.is_alive()


def test_sdk_roi_and_sensor_frame_rate():
    sdk, calls, _ = fake_sdk()
    cfg = config(sensor_roi=dict(enabled=True, offset_x=180, offset_y=0, width=1080, height=1080), exposure_time=5000)
    cap = MvsCapture(cfg, sdk=sdk).open()
    cap.close()
    # ROI is deliberately applied in software after mounting rotation.
    assert ('MV_CC_SetIntValue', ('Width', 4)) in calls
    assert ('MV_CC_SetIntValue', ('Height', 4)) in calls
    assert ('MV_CC_SetIntValue', ('OffsetX', 0)) in calls
    assert ('MV_CC_SetFloatValue', ('AcquisitionFrameRate', 30.0)) in calls
    assert ('MV_CC_SetFloatValue', ('ExposureTime', 5000.0)) in calls


def test_example_profile_is_valid():
    import yaml
    profile = Path(__file__).resolve().parents[1] / 'example/eval_robots_config_mvs.yaml'
    cfg = yaml.safe_load(profile.read_text())
    camera = normalize_mvs_configs([cfg['mvs']['right_hand']])[0]
    assert camera['serial'] == 'DA9057815'
    assert camera['receive_latency'] == 0.0
    assert camera['fps'] == 30


def test_environment_checks_mvs_before_starting_robot_workers():
    import ast
    source = Path(__file__).resolve().parents[1] / 'umi/real_world/bimanual_umi_env.py'
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'BimanualUmiEnv')
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == 'start')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), scope)
    calls = []
    def failed_camera(wait):
        assert wait is True
        raise RuntimeError('missing SDK')
    robot = SimpleNamespace(start=lambda **kw: calls.append('robot'))
    env = SimpleNamespace(camera_startup_first=True, camera=SimpleNamespace(start=failed_camera),
                          robots=[robot], grippers=[], multi_cam_vis=None)
    with pytest.raises(RuntimeError, match='missing SDK'):
        scope['start'](env)
    assert calls == []

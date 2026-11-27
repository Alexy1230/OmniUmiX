"""Collection-compatible Rectify -> archived runtime Solver -> RGB Difference."""
import argparse
import atexit
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import uuid
import unicodedata

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from umi.real_world.mvs_camera_service import _send, _recv


FINGERS = ('left', 'right')  # Fingers of the RIGHT gripper, not robot arm order.


def validate_config(config):
    config = dict(config)
    for side in FINGERS:
        serial = config.get(side, {}).get('serial')
        if not isinstance(serial, str) or not serial.strip() or serial.startswith('FILL_'):
            raise ValueError(f'policy.tactile.{side}.serial must identify the physical {side} finger')
    if config['left']['serial'] == config['right']['serial']:
        raise ValueError('Left and right fingers must have distinct Xense serials')
    for key, default in [('startup_timeout_s', 30.0), ('timeout_s', 5.0), ('max_stale_s', 1.0)]:
        value = float(config.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f'tactile.{key} must be finite and positive')
        config[key] = value
    config.setdefault('infer_mode', 'normal')
    config.setdefault('runtime_mode', 'calibrate')
    config.setdefault('record_force', False)
    if config['runtime_mode'] not in ('calibrate', 'saved'):
        raise ValueError('tactile.runtime_mode must be calibrate or saved')
    if not isinstance(config['record_force'], bool):
        raise ValueError('tactile.record_force must be a YAML boolean')
    for side in FINGERS:
        if config[side].get('config_path'):
            raise ValueError('Use runtime_path for createSolver; online Sensor config_path is not the collection path')
        if config['runtime_mode'] == 'saved':
            path = config[side].get('runtime_path')
            if not path:
                raise ValueError(f'tactile.{side}.runtime_path is required in saved mode')
            if not Path(path).expanduser().is_file():
                raise FileNotFoundError(f'Xense {side} runtime file not found: {path}')
    return config


def confirm_unloaded_calibration():
    """Require deliberate confirmation; a stray Enter/typo must not abort startup."""
    print('[Calibration] Remove any object and keep BOTH tactile pads completely unloaded.', flush=True)
    while True:
        try:
            value = input('[Calibration] Type calibrate then Enter to continue; q to quit: ')
        except EOFError as error:
            raise RuntimeError(
                'Calibration confirmation needs an interactive terminal. Run the client in a terminal '
                '(or use ssh -t); runtime_mode: saved can reuse already verified runtime files.'
            ) from error
        command = unicodedata.normalize('NFKC', value).strip().casefold()
        if command == 'calibrate':
            print('[Calibration] Confirmation accepted; starting Xense calibration service.', flush=True)
            return
        if command in ('q', 'quit', 'cancel'):
            raise RuntimeError('Xense calibration cancelled by user; SDK and robot have not started')
        print('[Calibration] Still waiting: type the full word calibrate and press Enter. '
              'No calibration or robot startup has occurred.', flush=True)


class XenseCapture:
    def __init__(self, config):
        self.config = validate_config(config)
        self.process = self.directory = self.address = None
        self.owner_pid = os.getpid()

    def _log(self):
        path = Path(self.directory) / 'service.log' if self.directory else None
        return path.read_text(errors='replace')[-8000:] if path and path.exists() else ''

    def _rpc(self, method):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(self.config['timeout_s'])
            conn.connect(self.address)
            _send(conn, method)
            ok, result = _recv(conn)
        if not ok:
            raise RuntimeError(f'Xense {method}: {result}')
        return result

    def start(self):
        if self.process is not None:
            raise RuntimeError('Xense capture already started')
        if self.config['runtime_mode'] == 'calibrate':
            confirm_unloaded_calibration()
        self.directory = tempfile.mkdtemp(prefix='umi-xense-', dir='/tmp')
        self.address = str(Path(self.directory) / 'sensor.sock')
        path = Path(self.directory) / 'config.json'
        path.write_text(json.dumps(self.config))
        command = [self.config.get('python') or sys.executable, '-u', str(Path(__file__).resolve()),
                   '--config', str(path), '--socket', self.address, '--parent-pid', str(self.owner_pid)]
        try:
            with (Path(self.directory) / 'service.log').open('wb') as log:
                self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                                stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + self.config['startup_timeout_s']
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError('Xense startup failed:\n' + self._log())
                if os.path.exists(self.address):
                    self._rpc('ping')
                    atexit.register(self.close)
                    print(f'Xense ready (PID {self.process.pid}); fingers left/right configured explicitly', flush=True)
                    return self
                time.sleep(0.05)
            raise TimeoutError('Xense startup timed out:\n' + self._log())
        except BaseException:
            self.close()
            raise

    def read(self):
        try:
            return self._rpc('read')
        except (OSError, EOFError) as error:
            raise RuntimeError(f'Xense acquisition failed: {error}\n{self._log()}') from error

    def close(self):
        if os.getpid() != self.owner_pid:
            return
        atexit.unregister(self.close)
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    self._rpc('stop')
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=2)
            self.process = None
        if self.directory:
            shutil.rmtree(self.directory, ignore_errors=True)
        self.directory = self.address = None


class SensorPair:
    def __init__(self, config, sensor_class=None):
        if sensor_class is None:
            # Same ordering as v21/cameras/xense/camera_xense.py.
            import onnxruntime as ort
            ort.preload_dlls(directory="")
            from xensesdk import Sensor as sensor_class
        self.sdk, self.config = sensor_class, validate_config(config)
        self.sensors, self.solvers = {}, {}
        self.previous, self.runtime_info = {}, {}
        try:
            devices = self.sdk.scanSerialNumber()
            for side in FINGERS:
                spec = self.config[side]
                if spec['serial'] not in devices:
                    raise RuntimeError(f'Xense {side} serial {spec["serial"]!r} absent; available={devices}')
                # Match XenseCamera.connect(): online Sensor only acquires Rectify.
                self.sensors[side] = self.sdk.create(cam_id=devices[spec['serial']],
                                                     infer_mode=self.config['infer_mode'])
            # Collection connect(warmup=True) waits for a valid Rectify before
            # the operator-triggered calibrate/export stage.
            deadline = time.monotonic() + min(10.0, self.config['startup_timeout_s'])
            for side in FINGERS:
                while True:
                    try:
                        self._read_rectify(side)
                        break
                    except (ValueError, RuntimeError, TypeError):
                        if time.monotonic() >= deadline:
                            raise
                        time.sleep(0.05)
            if self.config['runtime_mode'] == 'calibrate':
                root = Path(self.config.get('runtime_directory') or 'data_local/xense_runtime').expanduser().resolve()
                root.mkdir(parents=True, exist_ok=True)
                runtime_dir = root / f'xrs_{uuid.uuid4().hex}'
                runtime_dir.mkdir()
            for side, sensor in self.sensors.items():
                spec = self.config[side]
                if self.config['runtime_mode'] == 'calibrate':
                    # XenseCamera.export_runtime_snapshot(): no application read
                    # thread runs here, so calibrate/export cannot race our reads.
                    sensor.calibrateSensor()
                    sensor.exportRuntimeConfig(save_dir=str(runtime_dir), binary=False)
                    runtime = runtime_dir / f'runtime_{spec["serial"]}'
                else:
                    runtime = Path(spec['runtime_path']).expanduser().resolve()
                if not runtime.is_file():
                    raise FileNotFoundError(f'Xense runtime export/file missing: {runtime}')
                self.runtime_info[side] = dict(serial=spec['serial'], runtime_path=str(runtime),
                                              sha256=hashlib.sha256(runtime.read_bytes()).hexdigest())
                self.solvers[side] = self.sdk.createSolver(runtime)
                if not self.solvers[side]:
                    raise RuntimeError(f'Xense SDK could not create solver from {runtime}')
            if self.config['runtime_mode'] == 'calibrate':
                try:
                    version = importlib.metadata.version('xensesdk')
                except importlib.metadata.PackageNotFoundError:
                    version = 'unknown'
                # Keep the collection runtime-set schema and logical sensor names.
                sensors = {f'right_hand_{side}_finger': dict(
                    serial=info['serial'], runtime=Path(info['runtime_path']).name,
                    runtime_sha256=info['sha256']) for side, info in self.runtime_info.items()}
                (runtime_dir/'manifest.json').write_text(json.dumps(dict(
                    schema_version=2, runtime_set_id=runtime_dir.name,
                    reason='deployment_unloaded_calibration', sdk_version=version,
                    sensors=sensors, created_at=time.time()), indent=2))
            deadline = time.monotonic() + min(10.0, self.config['startup_timeout_s'])
            while True:
                try:
                    self.read()
                    break
                except (ValueError, RuntimeError, TypeError):
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def _read_rectify(self, side):
        import numpy as np
        rectify, stamp = self.sensors[side].selectSensorInfo(
            self.sdk.OutputType.Rectify, self.sdk.OutputType.TimeStamp)
        rectify, stamp = np.asarray(rectify), float(stamp)
        if rectify.shape != (700, 400, 3) or rectify.dtype != np.uint8:
            raise ValueError(f'Xense {side} Rectify must be uint8 [700,400,3], got {rectify.shape}/{rectify.dtype}')
        if not math.isfinite(stamp):
            raise ValueError(f'Xense {side} timestamp must be finite')
        return rectify, stamp

    def read(self):
        import numpy as np
        result = {}
        for side in FINGERS:
            started = time.time()
            rectify, stamp = self._read_rectify(side)
            now = time.monotonic()
            old = self.previous.get(side)
            if old and stamp == old[0]:
                if now - old[1] > self.config['max_stale_s']:
                    raise RuntimeError(f'Xense {side} timestamp stopped advancing')
            else:
                self.previous[side] = stamp, now
            outputs = [self.sdk.OutputType.Difference]
            if self.config['record_force']:
                outputs.append(self.sdk.OutputType.ForceResultant)
            # Pipeline 09 reads BGR Rectify with cv2.VideoCapture and passes it
            # unchanged to Solver. Do not pass the RGB observation here.
            values = self.solvers[side].selectSensorInfo(*outputs, rectify_image=rectify)
            force = None
            if self.config['record_force']:
                image, force = values
                force = np.asarray(force, dtype=np.float32)
                if force.shape != (6,) or not np.isfinite(force).all():
                    raise ValueError(f'Xense {side} requires finite ForceResultant[6]')
            else:
                image = values[0] if isinstance(values, (tuple, list)) and len(values) == 1 else values
            image = np.asarray(image)
            if image.shape != (700, 400, 3) or image.dtype != np.uint8:
                raise ValueError(f'Xense {side} Difference must be uint8 [700,400,3]')
            result[side] = dict(image_rgb=np.ascontiguousarray(image[..., ::-1]),
                                sensor_timestamp=stamp, read_started_at=started, received_at=time.time(),
                                runtime=self.runtime_info[side].copy())
            if force is not None:
                result[side]['force_resultant'] = force.copy()
        return result

    def close(self):
        for resource in [*self.solvers.values(), *self.sensors.values()]:
            try:
                if hasattr(resource, 'release'):
                    resource.release()
            except Exception:
                traceback.print_exc()
        self.solvers.clear()
        self.sensors.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--parent-pid', required=True, type=int)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    pair = SensorPair(config)
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(args.socket)
            server.listen(4)
            server.settimeout(1.0)
            while True:
                if os.getppid() != args.parent_pid:
                    break
                try:
                    conn, _ = server.accept()
                except socket.timeout:
                    continue
                with conn:
                    conn.settimeout(config.get('timeout_s', 5.0))
                    try:
                        method = _recv(conn)
                        if method == 'stop':
                            _send(conn, (True, None))
                            break
                        if method not in ('read', 'ping'):
                            raise ValueError(f'Unknown Xense method: {method}')
                        _send(conn, (True, pair.read() if method == 'read' else True))
                    except Exception:
                        _send(conn, (False, traceback.format_exc()))
    finally:
        pair.close()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    main()

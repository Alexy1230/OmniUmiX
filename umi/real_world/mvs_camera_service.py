"""Isolated MVS service and local proxy; no SDK/robot imports in the proxy.

The executable service owns the complete existing MultiMvsCamera stack. A
private Unix socket transfers numpy observations without lossy compression.
Never expose this pickle-based local protocol as a network service.
"""
import argparse
import atexit
import json
import math
import os
from pathlib import Path
import pickle
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import time
import traceback

_MAX_MESSAGE = 256 * 1024 * 1024


def _send(sock, value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > _MAX_MESSAGE:
        raise ValueError('Camera IPC message exceeds 256 MiB')
    sock.sendall(struct.pack('!Q', len(payload)) + payload)


def _recv_exact(sock, count):
    data = bytearray()
    while len(data) < count:
        chunk = sock.recv(min(count - len(data), 1024 * 1024))
        if not chunk:
            raise EOFError('Camera service disconnected')
        data.extend(chunk)
    return data


def _recv(sock):
    size, = struct.unpack('!Q', _recv_exact(sock, 8))
    if size > _MAX_MESSAGE:
        raise ValueError('Camera IPC message exceeds 256 MiB')
    return pickle.loads(_recv_exact(sock, size))


def _copy_out(value, out):
    if out is None:
        return value
    for key, item in value.items():
        if isinstance(item, dict):
            out[key] = _copy_out(item, out.get(key))
        elif (key in out and hasattr(out[key], '__setitem__')
              and getattr(out[key], 'shape', None) == getattr(item, 'shape', None)):
            out[key][...] = item
        else:
            out[key] = item
    return out


class IsolatedMultiMvsCamera:
    """MultiUvcCamera-compatible proxy, including use by a forked visualizer."""
    def __init__(self, configs, obs_resolution, vis_resolution, get_max_k,
                 obs_float32=False, rpc_timeout=5.0, _service_script=None):
        if not configs:
            raise ValueError('At least one MVS camera is required')
        self.options = dict(configs=configs, obs_resolution=list(obs_resolution),
                            vis_resolution=list(vis_resolution), get_max_k=get_max_k,
                            obs_float32=obs_float32)
        # Include interpreter import time in addition to each camera's deadline.
        self.launch_timeout = 10 + sum(float(c.get('launch_timeout', 15)) for c in configs)
        self.rpc_timeout = float(rpc_timeout)
        if not math.isfinite(self.rpc_timeout) or self.rpc_timeout <= 0:
            raise ValueError('rpc_timeout must be finite and positive')
        self._script = str(_service_script or Path(__file__).resolve())
        self._process = None
        self._directory = None
        self._socket_path = None
        self._owner_pid = os.getpid()
        self._stopping = False

    @property
    def n_cameras(self):
        return len(self.options['configs'])

    def _log_tail(self):
        if not self._directory:
            return ''
        path = Path(self._directory) / 'service.log'
        return path.read_text(errors='replace')[-12000:] if path.exists() else ''

    def _rpc(self, method, **kwargs):
        if not self._socket_path:
            raise RuntimeError('Camera service is not started')
        try:
            # One connection per call: the policy and forked visualizer never
            # share a stream or race over replies. Directory permissions are 0700.
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(self.rpc_timeout)
                conn.connect(self._socket_path)
                _send(conn, (method, kwargs))
                ok, result = _recv(conn)
        except (OSError, EOFError) as exc:
            raise RuntimeError(f'Camera service {method} failed: {exc}\n{self._log_tail()}') from exc
        if not ok:
            raise RuntimeError(f'Camera service {method}:\n{result}')
        return result

    @property
    def is_ready(self):
        if not self._socket_path or self._stopping:
            return False
        try:
            return bool(self._rpc('is_ready'))
        except RuntimeError:
            return False

    def start(self, wait=True, put_start_time=None):
        if self._process is not None:
            raise RuntimeError('Camera service already started')
        self._owner_pid = os.getpid()
        self._stopping = False
        # Short path avoids the Unix socket path-length limit on macOS/Linux.
        self._directory = tempfile.mkdtemp(prefix='umi-mvs-', dir='/tmp')
        self._socket_path = str(Path(self._directory) / 'camera.sock')
        config_path = Path(self._directory) / 'config.json'
        config_path.write_text(json.dumps(dict(self.options, put_start_time=put_start_time)))
        env = os.environ.copy()
        for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'OMP_NUM_THREADS'):
            env[name] = '1'
        try:
            with (Path(self._directory) / 'service.log').open('wb') as log:
                self._process = subprocess.Popen(
                    [sys.executable, '-u', self._script, '--config', str(config_path),
                     '--socket', self._socket_path, '--parent-pid', str(self._owner_pid)],
                    env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            atexit.register(self.stop)
            if wait:
                self.start_wait()
        except BaseException:
            self.stop()
            raise

    def start_wait(self):
        deadline = time.monotonic() + self.launch_timeout
        try:
            while time.monotonic() < deadline:
                if self._process is None or self._process.poll() is not None:
                    raise RuntimeError(f'MVS service exited during startup:\n{self._log_tail()}')
                if os.path.exists(self._socket_path):
                    if self._rpc('is_ready'):
                        print(f'MVS service ready (PID {self._process.pid})', flush=True)
                        return
                time.sleep(0.05)
            raise TimeoutError(f'MVS service startup timed out:\n{self._log_tail()}')
        except BaseException:
            self.stop()
            raise

    def get(self, k=None, out=None):
        return _copy_out(self._rpc('get', k=k), out)

    def get_vis(self, out=None):
        return _copy_out(self._rpc('get_vis'), out)

    def restart_put(self, start_time):
        return self._rpc('restart_put', start_time=float(start_time))

    def start_recording(self, video_path, start_time):
        # Service is a separate program, so recording paths must be absolute.
        if isinstance(video_path, (str, Path)):
            directory = Path(video_path).resolve()
            directory.mkdir(parents=True, exist_ok=True)
            video_path = [str(directory / f'{i}.mp4') for i in range(self.n_cameras)]
        else:
            video_path = [str(Path(p).resolve()) for p in video_path]
        if len(video_path) != self.n_cameras:
            raise ValueError('One recording path is required per camera')
        return self._rpc('start_recording', video_path=video_path, start_time=float(start_time))

    def stop_recording(self):
        return self._rpc('stop_recording')

    def stop(self, wait=True):
        if os.getpid() != self._owner_pid:
            return  # A visualizer child must never kill the owner's service.
        if self._process and not self._stopping:
            self._stopping = True
            if self._process.poll() is None and os.path.exists(self._socket_path):
                try:
                    self._rpc('stop')
                except RuntimeError:
                    pass
        if wait:
            self.stop_wait()

    def stop_wait(self):
        if os.getpid() != self._owner_pid:
            return
        atexit.unregister(self.stop)
        if self._process:
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Kill the isolated process group, including SDK/recorder workers,
                # if a native call prevents normal cleanup.
                try:
                    os.killpg(self._process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self._process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    pass
            # Also reap surviving descendants after a crash of the service root.
            try:
                os.killpg(self._process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._process.wait(timeout=2)
        self._process = None
        if self._directory:
            shutil.rmtree(self._directory, ignore_errors=True)
        self._directory = self._socket_path = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()


def serve(camera, address, parent_pid, max_frame_age=5.0):
    """Serve only the camera contract, never arbitrary attribute/method access."""
    allowed = {'get', 'get_vis', 'restart_put', 'start_recording', 'stop_recording'}

    def healthy():
        if not camera.is_ready:
            return False
        # An SDK call can hang after startup while the process remains alive.
        # Do not keep serving an old frame as a healthy observation in that case.
        latest = camera.get()
        for frame in latest.values():
            stamp = float(frame.get('camera_receive_timestamp', frame['timestamp']))
            if not math.isfinite(stamp) or time.time() - stamp > max_frame_age:
                return False
        return True

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
        listener.bind(address)
        listener.listen(16)
        listener.settimeout(0.5)
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                return
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                if not healthy():
                    raise RuntimeError('Camera/recorder stopped or frames became stale')
                continue
            with conn:
                conn.settimeout(5)
                stopping = False
                try:
                    method, kwargs = _recv(conn)
                    if method == 'stop':
                        stopping, result = True, None
                    elif method == 'is_ready':
                        result = healthy()
                    elif method in allowed:
                        if not healthy():
                            raise RuntimeError('Camera/recorder stopped or frames became stale')
                        result = getattr(camera, method)(**kwargs)
                    else:
                        raise ValueError(f'Unsupported camera operation: {method}')
                    _send(conn, (True, result))
                except Exception:
                    error = traceback.format_exc()
                    print(error, flush=True)
                    try:
                        _send(conn, (False, error))
                    except (OSError, EOFError):
                        pass
                if stopping:
                    return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--parent-pid', required=True, type=int)
    args = parser.parse_args()
    options = json.loads(Path(args.config).read_text())
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root))
    # Imported only in this new interpreter, never through the policy's proxy.
    import cv2
    from multiprocessing.managers import SharedMemoryManager
    from umi.real_world.mvs_camera import MultiMvsCamera
    from umi.real_world.mvs_camera_config import normalize_mvs_configs, MvsImageTransform
    cv2.setNumThreads(1)
    configs = normalize_mvs_configs(options['configs'])
    print(f'MVS service PID {os.getpid()}: starting camera-only stack', flush=True)
    with SharedMemoryManager() as manager:
        camera = MultiMvsCamera(
            configs, shm_manager=manager, get_max_k=options['get_max_k'],
            transform=[MvsImageTransform(tuple(options['obs_resolution']), float32=options['obs_float32']) for _ in configs],
            vis_transform=[MvsImageTransform(tuple(options['vis_resolution']), rgb=False) for _ in configs],
        )
        try:
            camera.start(put_start_time=options.get('put_start_time'))
            print('First frame and recorder ready', flush=True)
            max_age = max(5.0, max(c['frame_timeout_ms'] for c in configs) / 1000 * 2)
            serve(camera, args.socket, args.parent_pid, max_frame_age=max_age)
        finally:
            camera.stop()


if __name__ == '__main__':
    main()

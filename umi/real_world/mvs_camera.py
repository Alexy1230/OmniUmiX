"""MVS workers using UMI's existing buffering, transforms and video recorder."""
import multiprocessing as mp
import time
import traceback

import numpy as np

from umi.real_world.mvs_capture import MvsCapture
from umi.real_world.uvc_camera import UvcCamera
from umi.real_world.multi_uvc_camera import MultiUvcCamera
from umi.real_world.video_recorder import VideoRecorder


class MvsCamera(UvcCamera):
    def __init__(self, config, **kwargs):
        self.mvs_config = dict(config)
        self.launch_timeout = float(config.get('launch_timeout', 15))
        self.error_reader, self.error_writer = mp.Pipe(duplex=False)
        super().__init__(dev_video_path=config['serial'], **kwargs)

    def _open_capture(self):
        return MvsCapture(self.mvs_config).open()

    def _read_capture(self, cap, frame):
        image, received = cap.read()
        np.copyto(frame, image)
        # No hardware exposure timestamp is exposed by this adapter. Both raw
        # timestamp fields are host receipt time; run() subtracts latency once.
        return frame, received, received

    def _close_capture(self, cap):
        cap.close()

    def run(self):
        try:
            super().run()
        except BaseException:
            self.error_writer.send(traceback.format_exc())
            raise
        finally:
            self.ready_event.clear()

    @property
    def is_ready(self):
        return (self.ready_event.is_set() and self.is_alive()
                and self.video_recorder.is_alive() and not self.stop_event.is_set())

    def start_wait(self):
        deadline = time.monotonic() + self.launch_timeout
        while time.monotonic() < deadline:
            if self.error_reader.poll():
                raise RuntimeError(self.error_reader.recv())
            if not self.is_alive() or not self.video_recorder.is_alive():
                raise RuntimeError(f'MVS camera {self.dev_video_path} or recorder exited during startup')
            if self.is_ready and self.video_recorder.ready_event.is_set():
                return
            self.ready_event.wait(0.05)
        raise TimeoutError(f'MVS camera {self.dev_video_path}: no valid frame within {self.launch_timeout}s')

    def stop(self, wait=True):
        # Stop the producer before shutting down the recorder so its queued
        # frames are flushed after the final camera frame.
        self.stop_event.set()
        if wait:
            self.end_wait()

    def end_wait(self):
        if self.pid is not None:
            self.join(timeout=self.launch_timeout)
            if self.is_alive():
                self.terminate()
                self.join()
        if self.video_recorder.is_started:
            self.video_recorder.stop()
            self.video_recorder.join(timeout=self.launch_timeout)
            if self.video_recorder.is_alive():
                self.video_recorder.terminate()
                self.video_recorder.join()


class MultiMvsCamera(MultiUvcCamera):
    """Same get/get_vis/recording contract as MultiUvcCamera, no ROS dependency."""
    def __init__(self, configs, shm_manager, transform, vis_transform, get_max_k):
        self.shm_manager = shm_manager
        self.cameras = {}
        for index, config in enumerate(configs):
            fps = config['fps']
            # CPU encoding works without a NVIDIA GPU; NVENC remains opt-in.
            codec = config.get('recording_codec', 'libx264')
            recorder = VideoRecorder.create_h264(
                fps=fps, codec=codec, input_pix_fmt='bgr24')
            self.cameras[config['serial']] = MvsCamera(
                config=config, shm_manager=shm_manager,
                resolution=(config['width'], config['height']), capture_fps=fps,
                receive_latency=config.get('receive_latency', 0.0),
                put_downsample=False, get_max_k=get_max_k,
                transform=transform[index], vis_transform=vis_transform[index],
                video_recorder=recorder)

    def start(self, wait=True, put_start_time=None):
        try:
            super().start(wait=wait, put_start_time=put_start_time)
        except BaseException:
            self.stop()
            raise

    def start_wait(self):
        try:
            super().start_wait()
        except BaseException:
            self.stop()
            raise

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.end_wait()

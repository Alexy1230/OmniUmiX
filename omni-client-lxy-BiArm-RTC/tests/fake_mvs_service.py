"""Subprocess fixture: exercise real IPC without SDK, serial or robot imports."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'umi/real_world'))
from mvs_camera_service import serve

class Camera:
    is_ready = True
    def __init__(self):
        self.start_time = time.time()
    def get(self, k=None):
        t = time.time() - (10 if getattr(self, 'stale', False) else 0)
        stamps = np.array([t]) if k is None else t - np.arange(k)[::-1] / 30
        color = np.full((4, 4, 3), 17, np.uint8)
        if k is not None:
            color = np.repeat(color[None], k, axis=0)
        return {0: {'color': color, 'timestamp': stamps[0] if k is None else stamps}}
    def get_vis(self):
        return {'color': np.full((1, 8, 8, 3), 23, np.uint8)}
    def restart_put(self, start_time):
        self.start_time = start_time
    def start_recording(self, video_path, start_time):
        for path in video_path:
            Path(path).write_text(str(start_time))
    def stop_recording(self):
        pass

p = argparse.ArgumentParser()
p.add_argument('--socket')
p.add_argument('--config')
p.add_argument('--parent-pid', type=int)
a = p.parse_args()
config = json.loads(Path(a.config).read_text())['configs'][0]
if config.get('fail'):
    raise RuntimeError('Simulated SDK open error')
if config.get('hang'):
    time.sleep(30)
camera = Camera()
if config.get('stale_after_start'):
    def get_vis():
        camera.stale = True
        return {'color': np.full((1, 8, 8, 3), 23, np.uint8)}
    camera.get_vis = get_vis
serve(camera, a.socket, a.parent_pid)

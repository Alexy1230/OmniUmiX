"""Lossless policy-input capture; one bounded background PNG writer."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
import time

import numpy as np


class PolicyInputRecorder:
    def __init__(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix='run_', dir=str(root)))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='tactile-png')
        self.pending = deque()
        self.closed = False
        self.reported_phases = set()
        print(f'Lossless tactile/model inputs: {self.directory}', flush=True)

    def submit(self, request_id, payload, tactile, settings, phase='policy', observation_timestamp=None):
        if self.closed:
            raise RuntimeError('Policy input recorder is closed')
        while self.pending and self.pending[0].done():
            self.pending.popleft().result()
        if len(self.pending) >= 4:
            print('[Capture] Waiting for PNG writer; frames will not be silently dropped.', flush=True)
            self.pending.popleft().result()
        # The camera/shared-memory buffers may be reused before the writer runs.
        images = {key: np.array(value, copy=True) for key, value in payload.items()
                  if key.startswith('observation/image') or key.startswith('observation/tactile_')}
        sensors = {}
        for side, sample in (tactile or {}).items():
            sensors[side] = {key: (value.copy() if isinstance(value, np.ndarray) else value)
                             for key, value in sample.items()}
        metadata = dict(request_id=request_id, phase=phase, queued_at=time.time(),
                        observation_timestamp=observation_timestamp, prompt=payload.get('prompt'),
                        gripper_calibration=settings.get('gripper_calibration'),
                        state_format='absolute xyz metres, rotation first two columns, model gripper radians',
                        format='HWC uint8 RGB, lossless PNG', sensors={
                            side:{k:(v.tolist() if isinstance(v,np.ndarray) else v)
                                  for k,v in sample.items() if k!='image_rgb'}
                            for side,sample in sensors.items()})
        state = np.asarray(payload['current_tcp_pose9_gripper']).copy()
        self.pending.append(self.pool.submit(self._write, request_id, images, state, sensors, metadata))

    def _write(self, request_id, images, state, sensors, metadata):
        import cv2
        directory = self.directory/f'request_{request_id:06d}'
        directory.mkdir()
        files = {}

        def write(name, rgb):
            rgb = np.asarray(rgb)
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f'{name}: expected HWC uint8 RGB')
            if not cv2.imwrite(str(directory/name), np.ascontiguousarray(rgb[..., ::-1]),
                               [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                raise IOError(f'Cannot write {directory/name}')
            return name

        for side, sample in sensors.items():
            files[f'{side}_raw_difference'] = write(f'{side}_difference_rgb.png', sample['image_rgb'])
        for key, image in images.items():
            name = key.removeprefix('observation/').replace('/', '_') + '_rgb.png'
            files[key] = write(name, image)
        np.save(directory/'state.npy', state, allow_pickle=False)
        metadata['files'] = files
        (directory/'metadata.json').write_text(json.dumps(metadata, indent=2))
        phase = metadata.get('phase', 'policy')
        if phase not in self.reported_phases:
            self.reported_phases.add(phase)
            print(f'[Capture] Saved first {phase} tactile/model input images: {directory}', flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            while self.pending:
                self.pending.popleft().result()
        finally:
            self.pool.shutdown(wait=True)

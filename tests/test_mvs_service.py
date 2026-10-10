"""Real subprocess/IPC tests; no camera SDK or robot hardware is loaded."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'umi/real_world'))
from mvs_camera_service import IsolatedMultiMvsCamera

class ServiceTests(unittest.TestCase):
    def camera(self, **kwargs):
        return IsolatedMultiMvsCamera(
            [dict(serial='FAKE', **kwargs)], (4, 4), (8, 8), 10,
            _service_script=ROOT / 'tests/fake_mvs_service.py')

    def test_frames_recording_concurrent_preview_and_cleanup(self):
        with self.camera() as camera, tempfile.TemporaryDirectory() as output:
            directory = camera._directory
            process = camera._process
            self.assertTrue(camera.is_ready)
            obs = camera.get(k=3)
            self.assertEqual(obs[0]['color'].shape, (3, 4, 4, 3))
            self.assertEqual(obs[0]['color'].dtype, np.uint8)
            self.assertTrue(np.all(np.diff(obs[0]['timestamp']) > 0))
            self.assertIs(camera.get(k=3, out=obs), obs)
            single = camera.get()
            self.assertIs(camera.get(out=single), single)
            errors = []
            def read():
                try:
                    for _ in range(10):
                        self.assertEqual(camera.get_vis()['color'].shape, (1, 8, 8, 3))
                except BaseException as exc:
                    errors.append(exc)
            t = threading.Thread(target=read)
            t.start()
            for _ in range(10):
                camera.get()
            t.join()
            self.assertFalse(errors)
            camera.restart_put(time.time())
            camera.start_recording(output, time.time())
            camera.stop_recording()
            self.assertTrue((Path(output) / '0.mp4').exists())
        self.assertIsNotNone(process.poll())
        self.assertFalse(Path(directory).exists())
        camera.stop()

    def test_startup_error_is_forwarded(self):
        camera = self.camera(fail=True)
        with self.assertRaisesRegex(RuntimeError, 'Simulated SDK open error'):
            camera.start()
        self.assertIsNone(camera._directory)

    def test_hung_startup_is_bounded_and_cleaned_up(self):
        camera = self.camera(hang=True)
        camera.launch_timeout = 0.3
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            camera.start()
        self.assertLess(time.monotonic() - started, 10)
        self.assertIsNone(camera._process)

    def test_worker_death_fails_reads(self):
        with self.camera() as camera:
            camera._process.kill()
            camera._process.wait()
            self.assertFalse(camera.is_ready)
            with self.assertRaises(RuntimeError):
                camera.get()

    def test_stale_frames_are_not_reported_as_ready(self):
        with self.camera(stale_after_start=True) as camera:
            camera.get_vis()
            self.assertFalse(camera.is_ready)
            with self.assertRaisesRegex(RuntimeError, 'stale'):
                camera.get()

if __name__ == '__main__':
    unittest.main()

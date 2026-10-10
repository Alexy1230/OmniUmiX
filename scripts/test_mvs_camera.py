"""Camera-only MVS check: acquisition, timestamps, preview and optional MP4."""
import argparse
from pathlib import Path
import sys
import time
from multiprocessing.managers import SharedMemoryManager

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'example/eval_robots_config_mvs.yaml'))
    parser.add_argument('--seconds', type=float, default=5)
    parser.add_argument('--output', help='Directory for per-camera MP4s')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--isolated', action='store_true',
                        help='Use the fresh camera service used by BimanualUmiEnv')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    import cv2
    import numpy as np
    import yaml
    from umi.real_world.mvs_camera import MultiMvsCamera
    from umi.real_world.mvs_camera_config import normalize_mvs_configs, MvsImageTransform

    with open(args.config) as stream:
        configs = normalize_mvs_configs(yaml.safe_load(stream)['cameras'])
    observed = [[] for _ in configs]

    with SharedMemoryManager() as manager:
        if args.isolated:
            from umi.real_world.mvs_camera_service import IsolatedMultiMvsCamera
            camera = IsolatedMultiMvsCamera(
                configs, obs_resolution=(240, 240), vis_resolution=(480, 480), get_max_k=10)
        else:
            camera = MultiMvsCamera(
                configs, shm_manager=manager, get_max_k=10,
                transform=[MvsImageTransform((224, 224)) for _ in configs],
                vis_transform=[MvsImageTransform((480, 480), rgb=False) for _ in configs])
        
        with camera:
            if args.output:
                directory = Path(args.output).resolve()
                directory.mkdir(parents=True, exist_ok=True)
                camera.start_recording([str(directory / f'{i}.mp4') for i in range(len(configs))], time.time())

            deadline = time.monotonic() + args.seconds
            while time.monotonic() < deadline:
                if not camera.is_ready:
                    raise RuntimeError('Camera/recorder worker stopped; inspect its error output')
                for index, data in camera.get().items():
                    stamp = float(data['timestamp'])
                    if not observed[index] or stamp != observed[index][-1]:
                        observed[index].append(stamp)
                if args.show:
                    for index, frame in enumerate(camera.get_vis()['color']):
                        cv2.imshow(configs[index]['serial'], frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                time.sleep(0.002)
            if args.output:
                camera.stop_recording()
                
        cv2.destroyAllWindows() if args.show else None
    for config, timestamps in zip(configs, observed):
        gaps = np.diff(timestamps)
        if len(gaps) == 0 or np.any(gaps <= 0):
            raise RuntimeError(f"{config['serial']}: insufficient or non-monotonic frames")
        print(f"{config['serial']}: {len(timestamps)} unique frames, "
              f"{1/np.mean(gaps):.1f} FPS, max frame gap {np.max(gaps)*1000:.1f} ms")
    if args.output:
        print(f'Recordings saved in {Path(args.output).resolve()}')


if __name__ == '__main__':
    main()

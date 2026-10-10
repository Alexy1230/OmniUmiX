"""Save episode-client camera preprocessing output, without robot/gripper/policy startup.

Uses the same isolated MVS service, MvsImageTransform, and final client resize.
PNG files are lossless views; NPY files contain the RGB uint8 payload arrays.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def positive_float(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('must be positive and finite')
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', '--robot-config', dest='config',
                        default=str(ROOT / 'example/eval_pi05_lift_bottle_mvs.yaml'))
    parser.add_argument('--output', required=True, help='Parent directory; creates a unique run folder')
    parser.add_argument('--seconds', type=positive_float, default=10.0)
    parser.add_argument('--interval', type=positive_float, default=0.5,
                        help='Minimum seconds between saved frames (default: 0.5)')
    parser.add_argument('--obs-image-resolution', type=int,
                        help='Same override as the episode client; otherwise policy.image_resolution or 240')
    parser.add_argument('--camera-reorder', help='Same comma-separated camera order as the episode client')
    parser.add_argument('--show', action='store_true', help='Preview final images; Q exits')
    args = parser.parse_args(argv)

    # Lazy imports let --help work without the robot environment or MVS SDK.
    import cv2
    import numpy as np
    import yaml
    from umi.real_world.mvs_camera_config import normalize_mvs_configs
    from umi.real_world.mvs_camera_service import IsolatedMultiMvsCamera

    with Path(args.config).expanduser().open() as stream:
        config = yaml.safe_load(stream)
    resolution = (args.obs_image_resolution if args.obs_image_resolution is not None
                  else int(config.get('policy', {}).get('image_resolution', 240)))
    if resolution <= 0:
        parser.error('image resolution must be positive')
    order = None if args.camera_reorder is None else [int(x) for x in args.camera_reorder.split(',')]
    configs = normalize_mvs_configs(config['cameras'], order)
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_dir = Path(tempfile.mkdtemp(prefix='camera_', dir=output))
    metadata = dict(
        config_path=str(Path(args.config).expanduser().resolve()), cameras=configs,
        image_resolution=resolution, saved_format='HWC RGB uint8',
        preprocessing='MvsCapture 1440x1080 -> mounting rotation -> rounded-center software ROI + black padding -> capture resize -> '
                      'MvsImageTransform linear resize + BGR-to-RGB -> client INTER_AREA resize + uint8',
        note='Latest camera observations only; no robot synchronization, policy connection, or action execution. '
             'These are processed payload images, not full-resolution SDK frames.',
    )
    (run_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(f'Saving processed {resolution}x{resolution} images to {run_dir}', flush=True)
    print('Camera only. No robot/gripper initialization or policy-server connection.', flush=True)
    camera = IsolatedMultiMvsCamera(
        configs=configs, obs_resolution=(resolution, resolution),
        vis_resolution=(resolution, resolution), get_max_k=10, obs_float32=False,
    )
    counts = [0] * len(configs)
    last_stamps = [None] * len(configs)
    next_save = [0.0] * len(configs)
    try:
        camera.start()
        deadline = time.monotonic() + args.seconds
        with (run_dir / 'frames.jsonl').open('w') as log:
            while time.monotonic() < deadline:
                if not camera.is_ready:
                    raise RuntimeError('MVS camera service is no longer ready')
                for index, data in camera.get().items():
                    stamp = float(data['timestamp'])
                    now = time.monotonic()
                    if stamp == last_stamps[index] or now < next_save[index]:
                        continue
                    # Exactly the episode client's final image operation. The
                    # service has already performed MvsImageTransform.
                    image = cv2.resize(
                        data['color'], (resolution, resolution),
                        interpolation=cv2.INTER_AREA,
                    ).astype(np.uint8, copy=False)
                    stem = f'camera{index}_{counts[index]:06d}'
                    if not cv2.imwrite(str(run_dir / f'{stem}.png'), image[..., ::-1]):
                        raise IOError(f'Could not write {stem}.png')
                    np.save(run_dir / f'{stem}.npy', image, allow_pickle=False)
                    log.write(json.dumps(dict(
                        camera=index, serial=configs[index]['serial'], timestamp=stamp,
                        saved_at=time.time(), shape=list(image.shape),
                        png=f'{stem}.png', rgb_npy=f'{stem}.npy',
                    )) + '\n')
                    log.flush()
                    counts[index] += 1
                    last_stamps[index] = stamp
                    next_save[index] = now + args.interval
                    print(f'{stem}: RGB {image.shape}, timestamp={stamp:.6f}', flush=True)
                    if args.show:
                        cv2.imshow(f'camera{index}: policy input', image[..., ::-1])
                if args.show and (cv2.waitKey(1) & 0xff) in (ord('q'), ord('Q')):
                    break
                time.sleep(0.005)
    except KeyboardInterrupt:
        print('Capture interrupted; keeping saved images.')
    finally:
        camera.stop()
        if args.show:
            cv2.destroyAllWindows()
        print(f'Saved frames per camera: {counts}; folder: {run_dir}', flush=True)
    if not all(counts):
        raise RuntimeError('At least one camera produced no saved images')


if __name__ == '__main__':
    main()

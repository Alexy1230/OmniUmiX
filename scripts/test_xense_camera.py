"""Camera-only check: no robot/gripper imports or movement, optional FTP1 request."""
import argparse
import json
from pathlib import Path
import sys
import time
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--output', default='data_local/xense_check')
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--server-url', help='Optional: send real tactile + supplied scene/state; never execute actions')
    parser.add_argument('--scene-image', help='RGB scene saved as PNG/JPEG (decoded from BGR by OpenCV)')
    parser.add_argument('--state', help='NPY absolute pose10, gripper in calibrated model radians')
    args = parser.parse_args()
    import cv2
    import numpy as np
    import yaml
    from umi.real_world.xense_capture import XenseCapture
    from umi.real_world.bimanual_policy import PolicyClient
    cfg = yaml.safe_load(Path(args.config).read_text())
    options = dict(cfg['policy'])
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix='run_', dir=str(output)))
    print(f'Lossless tactile frames: {output}', flush=True)
    client = None
    if args.server_url:
        if not args.scene_image or not args.state:
            parser.error('--server-url requires --scene-image and --state; no made-up robot state is supplied')
        image = cv2.imread(args.scene_image)
        if image is None:
            parser.error('Cannot decode --scene-image')
        state = np.load(args.state, allow_pickle=False)
        options['capture_dir'] = str(output)
        client = PolicyClient(args.server_url, options.get('prompt', ''),
                              timeout=options.get('request_timeout_s', 30), policy_config=options)
        capture = client.tactile
    else:
        tactile = dict(options['tactile'])
        tactile['runtime_directory'] = tactile.get('runtime_directory') or str(output/'xense_runtime')
        capture = XenseCapture(tactile).start()
    try:
        deadline, count = time.monotonic() + args.seconds, 0
        while time.monotonic() < deadline:
            if client:
                action = client.call_server([image[..., ::-1]], state)
                sample = client.last_tactile
                print(f'Server actions={action.shape}; finger_force={None if client.last_finger_force is None else client.last_finger_force.shape}')
            else:
                sample = capture.read()
            frame_dir = output/f'frame_{count:06d}'
            frame_dir.mkdir()
            frame_metadata = {}
            for side, data in sample.items():
                rgb = data['image_rgb']
                if not cv2.imwrite(str(frame_dir/f'{side}_difference_rgb.png'), rgb[..., ::-1]):
                    raise IOError('Cannot save Difference preview')
                model = cv2.resize(rgb, (224,224), interpolation=cv2.INTER_LINEAR)
                if not cv2.imwrite(str(frame_dir/f'{side}_model224_rgb.png'), model[...,::-1]):
                    raise IOError('Cannot save model tactile preview')
                frame_metadata[side] = {k: v.tolist() if isinstance(v,np.ndarray) else v
                                        for k,v in data.items() if k != 'image_rgb'}
                if 'force_resultant' in data:
                    np.save(frame_dir/f'{side}_force.npy', data['force_resultant'], allow_pickle=False)
                print(f'{side}: image={rgb.shape}, stamp={data["sensor_timestamp"]}, runtime={data["runtime"]}')
                if args.show:
                    cv2.imshow(side + ' Difference', rgb[..., ::-1])
            (frame_dir/'metadata.json').write_text(json.dumps(frame_metadata, indent=2))
            count += 1
            if args.show and cv2.waitKey(1) & 0xff == ord('q'):
                break
            time.sleep(0.1)
        print(f'Completed {count} reads; no robot commands were sent.')
    finally:
        client.close() if client else capture.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

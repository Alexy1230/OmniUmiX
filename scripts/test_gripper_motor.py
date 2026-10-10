"""Standalone LK serial-gripper check. Read-only unless --move-mm is supplied.

Uses the same GripperController and motor.motor package as BimanualUmiEnv.
No camera, robot, policy server, enable command or zero reset in read-only mode.
"""
import argparse
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def positive(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError('must be finite and positive')
    return value


def read_feedback(controller):
    motor = controller.motor1
    motor.refresh()
    if getattr(motor, 'refresh_ok', None) is not True:
        raise RuntimeError('Motor refresh failed; check serial port, motor ID, wiring and power')
    values = tuple(float(v) for v in (motor.getPosition(), motor.getVelocity(), motor.getTorque()))
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError('Motor returned non-finite feedback')
    return values


def print_feedback(controller, values):
    angle, velocity, torque = values
    width = controller._angle_to_width(angle) * 1000
    print(f'angle={angle:+.5f} rad  driver_width={width:+.2f} mm  '
          f'velocity={velocity:+.4f} rad/s  torque={torque:+.3f} Nm', flush=True)


def wait_feedback(controller, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if controller.thread is None or not controller.thread.is_alive():
            raise RuntimeError('Gripper control thread stopped')
        if controller.timestamp_history:
            age = time.time() - controller.receive_latency - controller.timestamp_history[-1]
            if age < 0.5:
                values = (controller.current_position_rad, controller.current_vel, controller.current_torque_nm)
                if all(math.isfinite(float(v)) for v in values):
                    return values
        time.sleep(0.02)
    raise TimeoutError('No fresh valid gripper feedback')


def run_test(controller, args):
    # Constructor opens the port only. Verify actual replies before enabling.
    print_feedback(controller, read_feedback(controller))
    if args.move_mm is None:
        deadline = time.monotonic() + args.seconds
        while time.monotonic() < deadline:
            print_feedback(controller, read_feedback(controller))
            time.sleep(0.1)
        print('PASS: valid motor feedback; no motion/enable/zero commands sent.')
        return

    target = args.move_mm / 1000.0
    if not math.isfinite(target) or not 0 <= target <= controller.MOTOR_PHYSICAL_LIMIT_M:
        raise ValueError('Requested width is outside the driver range')
    initial = controller._angle_to_width(controller.motor1.getPosition())
    if abs(target - initial) * 1000 > args.max_travel_mm:
        raise ValueError(f'Target differs from current driver width by more than {args.max_travel_mm:g} mm')
    if not 0 < args.torque_limit <= controller.TORQUE_LIMIT_SINGLE:
        raise ValueError('Torque limit must be positive and at most the driver TORQUE_LIMIT_SINGLE')
    # The existing loop clips to this field; move_gripper's force argument alone
    # does not set that clip in this driver version.
    controller.TORQUE_LIMIT_SINGLE = args.torque_limit
    controller.target_position_rad_max_delta = (
        args.speed_mm_s / controller.MM_PER_SINGLE_MOTOR_RADIAN * controller.DT
    )
    started = False
    try:
        started = True  # Cleanup even if enable or startup fails partway through.
        controller.start(reset_zero=False)
        wait_feedback(controller, args.feedback_timeout)
        print(f'Moving to driver width {args.move_mm:g} mm; preserving motor zero.', flush=True)
        controller.move_gripper(target, force_limit_nm=args.torque_limit)
        deadline = time.monotonic() + args.seconds
        stable = 0
        while time.monotonic() < deadline:
            values = wait_feedback(controller, args.feedback_timeout)
            print_feedback(controller, values)
            reached = abs(controller._angle_to_width(values[0]) - target) * 1000 <= args.tolerance_mm
            stable = stable + 1 if reached else 0
            if stable >= 3:
                print('PASS: target reached on three successive feedback samples.')
                return
            time.sleep(0.1)
        raise TimeoutError('Target was not reached before --seconds expired')
    finally:
        if started:
            controller.stop()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'example/eval_pi05_lift_bottle_mvs.yaml'))
    parser.add_argument('--port', help='Override grippers[0].port1')
    parser.add_argument('--motor-id', type=int, help='Override grippers[0].motor1_id')
    parser.add_argument('--seconds', type=positive, default=5.0)
    parser.add_argument('--move-mm', type=float, help='Explicit motion request: absolute driver width in mm')
    parser.add_argument('--max-travel-mm', type=positive, default=10.0)
    parser.add_argument('--speed-mm-s', type=positive, default=5.0)
    parser.add_argument('--torque-limit', type=positive, default=0.3)
    parser.add_argument('--tolerance-mm', type=positive, default=1.0)
    parser.add_argument('--feedback-timeout', type=positive, default=2.0)
    args = parser.parse_args(argv)
    import yaml
    with open(args.config) as stream:
        config = yaml.safe_load(stream)['grippers'][0]
    if config.get('gripper_type') != 'custom' or config.get('mode', 'single') != 'single':
        parser.error('This test requires a custom gripper in single-motor mode')
    port = args.port or config['port1']
    motor_id = args.motor_id if args.motor_id is not None else config['motor1_id']
    from umi.real_world.gripper_controller_1_14 import GripperController
    print(f'Opening LK motor ID {motor_id} on {port}; '
          f'mode={"MOVE" if args.move_mm is not None else "READ ONLY"}', flush=True)
    controller = GripperController(mode='single', port1=port, motor1_id=motor_id)
    try:
        run_test(controller, args)
    finally:
        # Controller.stop() disables output but does not close the serial port.
        controller.motor1.ser.close()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Interrupted; connection closed.')
        sys.exit(130)
    except Exception as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        sys.exit(1)

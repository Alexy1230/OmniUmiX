"""LK low-torque homing: require observed closing travel before accepting a stop."""
import math
import time


def home_motor(motor, config, torque_limit=2.0):
    torque = float(config.get('torque_nm', -0.05))
    timeout = float(config.get('timeout_s', 2.0))
    threshold = float(config.get('velocity_threshold_rad_s', 0.05))
    count = config.get('stall_samples', 10)
    minimum = float(config.get('min_duration_s', 0.2))
    min_travel = float(config.get('min_travel_rad', 0.02))
    if (not all(math.isfinite(v) for v in (torque, timeout, threshold, minimum, min_travel))
            or not -torque_limit <= torque < 0 or timeout <= 0 or threshold <= 0
            or minimum < 0 or minimum >= timeout or min_travel <= 0 or isinstance(count, bool)
            or not isinstance(count, int) or count < 1):
        raise ValueError('Invalid gripper homing torque/timeout/stall settings')
    started = time.monotonic()
    stalled, contact_position, initial_position = 0, None, None
    closing_travel, valid_reads, last_log = 0.0, 0, -math.inf
    print(f'[Gripper home] BEGIN: closing torque={torque:+.3f} Nm, timeout={timeout:g}s, '
          f'required closing travel={min_travel:g} rad. Motor zero is unchanged until motion + stop.', flush=True)
    try:
        while time.monotonic() - started < timeout:
            motor.refresh()
            if getattr(motor, 'refresh_ok', False) is not True:
                motor.set_torque_nm(0)
                stalled = 0
                time.sleep(0.01)
                continue
            position, velocity = float(motor.getPosition()), float(motor.getVelocity())
            if not math.isfinite(position) or not math.isfinite(velocity):
                raise RuntimeError('Non-finite motor feedback during homing')
            valid_reads += 1
            if initial_position is None:
                initial_position = position
            closing_travel = initial_position - position
            if closing_travel < -min_travel:
                raise RuntimeError('Gripper moved in the opening direction under closing torque; motor zero NOT changed')
            motor.set_torque_nm(torque)
            elapsed = time.monotonic() - started
            if elapsed-last_log >= 0.25:
                measured = float(motor.getTorque()) if hasattr(motor,'getTorque') else math.nan
                print(f'[Gripper home] t={elapsed:.2f}s angle={position:+.5f} rad '
                      f'vel={velocity:+.4f} rad/s requested_tau={torque:+.3f} Nm '
                      f'measured_tau={measured:+.3f} Nm closing_travel={closing_travel:.5f} rad '
                      f'stall={stalled}/{count}', flush=True)
                last_log = elapsed
            if (closing_travel >= min_travel and elapsed >= minimum and abs(velocity) < threshold):
                stalled += 1
            else:
                stalled = 0
            if stalled >= count:
                contact_position = position
                break
            time.sleep(0.01)
    finally:
        motor.set_torque_nm(0)
    if contact_position is None:
        if not valid_reads:
            detail = 'no valid motor feedback'
        elif closing_travel < min_travel:
            detail = ('no confirmed closing motion; it may already be closed, the torque may be insufficient, '
                      'or torque commands may not be taking effect. Start unloaded and slightly open')
        else:
            detail = 'closing motion occurred but no stable stop was confirmed before timeout'
        raise TimeoutError(f'Gripper homing failed: {detail}; closing_travel={closing_travel:.5f} rad. '
                           'Motor zero was NOT changed')
    time.sleep(0.5)
    motor.set_zero_ram()
    time.sleep(2)
    motor.refresh()
    if getattr(motor, 'refresh_ok', False) is not True:
        raise RuntimeError('No valid feedback after setting gripper zero')
    zero = float(motor.getPosition())
    if not math.isfinite(zero) or abs(zero) > 0.02:
        raise RuntimeError(f'Gripper zero verification failed: angle={zero} rad')
    print(f'[Gripper home] stop={contact_position:.5f} rad -> verified zero={zero:.5f} rad; '
          f'closing_travel={closing_travel:.5f} rad', flush=True)
    return contact_position

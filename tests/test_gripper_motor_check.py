"""No physical device: exercise test logic and the actual modified control loop."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('motor_check', ROOT / 'scripts/test_gripper_motor.py')
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

class FakeMotor:
    def __init__(self, *args, **kwargs):
        self.position, self.velocity, self.torque = 1.0, 0.0, 0.0
        self.refresh_ok = True
        self.calls = []
        self.ser = SimpleNamespace(close=lambda: self.calls.append('close'))
    def refresh(self):
        self.calls.append('refresh')
    def getPosition(self): return self.position
    def getVelocity(self): return self.velocity
    def getTorque(self): return self.torque
    def enable(self): self.calls.append('enable')
    def disable(self): self.calls.append('disable')
    def stop(self): self.calls.append('stop')
    def set_zero_ram(self): self.calls.append('zero')
    def set_torque_nm(self, value): self.calls.append(('torque', value))

motor = ModuleType('motor.motor')
motor.LkMotor = FakeMotor
protocol = ModuleType('motor.protocol')
protocol.radian_to_degree = lambda x: x * 180 / 3.1415926
logger = ModuleType('loguru')
logger.logger = SimpleNamespace(info=lambda *a: None)
spec = importlib.util.spec_from_file_location('real_controller', ROOT / 'umi/real_world/gripper_controller_1_14.py')
driver = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'motor': ModuleType('motor'), 'motor.motor': motor,
                             'motor.protocol': protocol, 'loguru': logger}):
    spec.loader.exec_module(driver)

class Tests(unittest.TestCase):
    def controller(self):
        return driver.GripperController(mode='single', port1='FAKE', motor1_id=2)
    def args(self, **kwargs):
        values = dict(move_mm=None, seconds=.001, max_travel_mm=10, torque_limit=.3,
                      speed_mm_s=5, feedback_timeout=1, tolerance_mm=1)
        values.update(kwargs)
        return SimpleNamespace(**values)
    def test_read_only_does_not_enable_move_or_zero(self):
        c = self.controller()
        check.run_test(c, self.args())
        self.assertTrue(c.motor1.calls)
        self.assertEqual(set(c.motor1.calls), {'refresh'})
    def test_invalid_feedback_fails(self):
        c = self.controller()
        c.motor1.refresh_ok = False
        with self.assertRaises(RuntimeError):
            check.run_test(c, self.args())
        self.assertNotIn('enable', c.motor1.calls)
    def test_far_target_rejected_before_enable(self):
        c = self.controller()
        with self.assertRaises(ValueError):
            check.run_test(c, self.args(move_mm=90))
        self.assertNotIn('enable', c.motor1.calls)
    def test_nonfinite_target_rejected(self):
        with self.assertRaises(ValueError):
            check.run_test(self.controller(), self.args(move_mm=float('nan')))
    def test_start_can_preserve_zero_and_default_retains_old_behavior(self):
        for reset in (False, True):
            c = self.controller()
            c._run = lambda: None
            with patch.object(driver.time, 'sleep', lambda _: None):
                c.start(reset_zero=reset)
            c.thread.join()
            self.assertEqual('zero' in c.motor1.calls, reset)
            c.stop()
    def test_first_target_ramp_begins_at_measured_position(self):
        c = self.controller()
        c.current_position_rad = 1.0
        c.motor1.position = 1.0
        c.move_gripper(.030, .3)  # 1 rad = 30 mm, already at target
        def torque(value):
            c.motor1.calls.append(('torque', value))
            c.command_queue.put(('shutdown',))
        c.motor1.set_torque_nm = torque
        c._single_control_loop()
        sent = [x[1] for x in c.motor1.calls if isinstance(x, tuple)]
        self.assertEqual(sent, [0.0])

if __name__ == '__main__':
    unittest.main()

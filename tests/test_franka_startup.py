"""Franka startup handshake tests; no robot or RPC server is contacted."""
import numpy as np
import zerorpc

from umi.real_world.franka_interpolation_controller import wait_for_cartesian_controller


def test_wait_for_cartesian_controller_retries_startup_race():
    class Robot:
        attempts = 0
        accepted = None

        def update_desired_ee_pose(self, pose):
            self.attempts += 1
            if self.attempts < 3:
                raise zerorpc.exceptions.RemoteError(
                    'Polymetis',
                    'Tried to perform a controller update with no controller running.',
                    'server traceback',
                )
            self.accepted = np.asarray(pose)

    robot = Robot()
    target = np.arange(6, dtype=float)
    wait_for_cartesian_controller(robot, target, timeout_s=1.0)

    assert robot.attempts == 3
    np.testing.assert_array_equal(robot.accepted, target)

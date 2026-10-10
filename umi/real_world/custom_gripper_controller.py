import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.real_world.gripper_controller_1_14 import GripperController


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2


class CustomGripperController(mp.Process):
    """
    Adapter that wraps gripper_controller_1.14.py to match UMI's gripper interface.

    This controller runs in a separate process and uses shared memory for IPC,
    matching the architecture of WSGController and FrankaGripperController.
    """

    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 port1: str,
                 motor1_id: int,
                 port2: str = None,
                 motor2_id: int = None,
                 mode: str = 'single',
                 frequency: int = 30,
                 move_max_speed: float = 0.2,
                 move_max_force: float = 10.0,
                 get_max_k: int = None,
                 command_queue_size: int = 1024,
                 launch_timeout: float = 3,
                 receive_latency: float = 0.0,
                 verbose: bool = False):
        """
        Initialize custom gripper controller.

        Args:
            shm_manager: SharedMemoryManager for IPC
            port1: Serial port for motor 1 (e.g., '/dev/ttyUSB0')
            motor1_id: Motor 1 ID
            port2: Serial port for motor 2 (for dual mode)
            motor2_id: Motor 2 ID (for dual mode)
            mode: 'single' or 'dual'
            frequency: Control loop frequency (Hz)
            move_max_speed: Maximum gripper speed (m/s)
            move_max_force: Maximum gripper force (Nm)
            get_max_k: Ring buffer size (default: frequency * 10)
            command_queue_size: Command queue size
            launch_timeout: Timeout for controller startup
            receive_latency: Latency compensation for observations
            verbose: Enable verbose logging
        """
        super().__init__(name="CustomGripperController")

        self.port1 = port1
        self.motor1_id = motor1_id
        self.port2 = port2
        self.motor2_id = motor2_id
        self.mode = mode
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.move_max_force = move_max_force
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)

        # Build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )

        # Build ring buffer
        example = {
            'gripper_state': 0,
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_measure_timestamp': time.time(),
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= Launch methods ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[CustomGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {'cmd': Command.SHUTDOWN.value}
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= Context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= Command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        """
        Schedule gripper to reach position at target_time.

        Args:
            pos: Target gripper width in meters
            target_time: Absolute timestamp when gripper should reach position
        """
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })

    # ========= Receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= Main loop in process ============
    def run(self):
        """Main control loop running in separate process."""
        try:
            # Initialize your gripper controller
            gripper = GripperController(
                mode=self.mode,
                port1=self.port1,
                motor1_id=self.motor1_id
            )

            # Start gripper controller
            gripper.start()
            time.sleep(1.0)  # Wait for initialization

            # Get initial position
            curr_pos = gripper.get_current_gripper_width()
            curr_t = time.monotonic()
            last_waypoint_time = curr_t

            # Initialize trajectory interpolator
            pose_interp = PoseTrajectoryInterpolator(
                times=[curr_t],
                poses=[[curr_pos, 0, 0, 0, 0, 0]]
            )

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0

            while keep_running:
                loop_start = time.perf_counter()

                # Calculate target position from interpolator
                t_now = time.monotonic()
                dt = 1 / self.frequency
                t_target = t_now
                target_pos = pose_interp(t_target)[0]

                # Send command to gripper
                if self.mode == 'single':
                    gripper.move_gripper(
                        width_m=target_pos,
                        force_limit_nm=self.move_max_force
                    )

                # Get state from gripper
                states = gripper.get_current_gripper_states()
                # states format: [width, vel, force, 0, 0, 0]

                # Convert to UMI state format
                state = {
                    'gripper_state': 0,  # 0 = success
                    'gripper_position': states[0],  # width in meters
                    'gripper_velocity': states[1],  # velocity in m/s
                    'gripper_force': states[2],     # force in Nm
                    'gripper_measure_timestamp': time.time(),
                    'gripper_receive_timestamp': time.time(),
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                # Fetch commands from queue
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Execute commands
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pos = command['target_pos']
                        target_time = command['target_time']
                        # Translate global time to monotonic time
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now

                        # Update trajectory interpolator
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=[target_pos, 0, 0, 0, 0, 0],
                            time=target_time,
                            max_pos_speed=self.move_max_speed,
                            max_rot_speed=self.move_max_speed,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    elif cmd == Command.RESTART_PUT.value:
                        t_start = command['target_time'] - time.time() + time.monotonic()
                        iter_idx = 1
                    else:
                        keep_running = False
                        break

                # First loop successful, ready to receive commands
                if iter_idx == 0:
                    self.ready_event.set()
                iter_idx += 1

                # Regulate frequency
                dt = 1 / self.frequency
                t_end = t_start + dt * iter_idx
                precise_wait(t_end=t_end, time_func=time.monotonic)

        except Exception as e:
            if self.verbose:
                print(f"[CustomGripperController] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Clean up
            try:
                gripper.stop()
            except:
                pass
            self.ready_event.set()
            if self.verbose:
                print(f"[CustomGripperController] Controller stopped")

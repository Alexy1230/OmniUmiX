"""
Teleop Interface - 启动机器人和夹爪控制器（无 ROS 依赖）

Usage:
    python teleop_interface.py

这个脚本启动：
1. FrankaInterpolationController - 通过 ZeroRPC 连接到 NUC 上的 Polymetis 服务器
2. CustomGripperController - 通过串口控制夹爪
"""
import os
import sys
import time
import signal
import psutil
from multiprocessing.managers import SharedMemoryManager

# 限制线程数
os.environ["OPENBLAS_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
os.environ["NUMEXPR_NUM_THREADS"] = "12"
os.environ["OMP_NUM_THREADS"] = "12"

import cv2
cv2.setNumThreads(12)

# 设置 CPU 亲和性（可选）
total_cores = psutil.cpu_count()
num_cores_to_bind = 8
cores_to_bind = set(range(total_cores - num_cores_to_bind, total_cores))
try:
    os.sched_setaffinity(0, cores_to_bind)
except Exception as e:
    print(f"Warning: Could not set CPU affinity: {e}")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from umi.real_world.franka_interpolation_controller import FrankaInterpolationController
from umi.real_world.custom_gripper_controller import CustomGripperController


# ============ 配置 ============
# Franka 机器人配置（NUC 上的 ZeroRPC 服务器）
FRANKA_IP = "172.16.0.3"
FRANKA_PORT = 4242
FRANKA_FREQUENCY = 200  # Hz

# 夹爪配置
GRIPPER_PORT = "/dev/ttyUSB2"
GRIPPER_MOTOR_ID = 1
GRIPPER_MODE = "single"
GRIPPER_FREQUENCY = 30  # Hz
GRIPPER_MAX_SPEED = 0.2  # m/s
GRIPPER_MAX_FORCE = 10.0  # Nm


class TeleopInterface:
    """
    遥操作接口 - 管理机器人和夹爪控制器
    """
    def __init__(self,
                 robot_ip=FRANKA_IP,
                 robot_port=FRANKA_PORT,
                 robot_frequency=FRANKA_FREQUENCY,
                 gripper_port=GRIPPER_PORT,
                 gripper_motor_id=GRIPPER_MOTOR_ID,
                 gripper_mode=GRIPPER_MODE,
                 gripper_frequency=GRIPPER_FREQUENCY,
                 verbose=True):

        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.robot_frequency = robot_frequency
        self.gripper_port = gripper_port
        self.gripper_motor_id = gripper_motor_id
        self.gripper_mode = gripper_mode
        self.gripper_frequency = gripper_frequency
        self.verbose = verbose

        self.shm_manager = None
        self.robot = None
        self.gripper = None
        self._running = False

    def start(self):
        """启动所有控制器"""
        print("=" * 50)
        print("Starting Teleop Interface (No ROS)")
        print("=" * 50)

        # 创建共享内存管理器
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        print("[TeleopInterface] SharedMemoryManager started")

        # 启动 Franka 控制器
        print(f"[TeleopInterface] Starting FrankaInterpolationController...")
        print(f"  Robot IP: {self.robot_ip}:{self.robot_port}")
        print(f"  Frequency: {self.robot_frequency} Hz")

        self.robot = FrankaInterpolationController(
            shm_manager=self.shm_manager,
            robot_ip=self.robot_ip,
            robot_port=self.robot_port,
            frequency=self.robot_frequency,
            Kx_scale=1.0,
            Kxd_scale=2.0,
            verbose=self.verbose,
            receive_latency=0.0
        )
        self.robot.start(wait=True)
        print("[TeleopInterface] FrankaInterpolationController started")

        # 启动夹爪控制器
        print(f"[TeleopInterface] Starting CustomGripperController...")
        print(f"  Port: {self.gripper_port}")
        print(f"  Motor ID: {self.gripper_motor_id}")
        print(f"  Mode: {self.gripper_mode}")

        self.gripper = CustomGripperController(
            shm_manager=self.shm_manager,
            port1=self.gripper_port,
            motor1_id=self.gripper_motor_id,
            mode=self.gripper_mode,
            frequency=self.gripper_frequency,
            move_max_speed=GRIPPER_MAX_SPEED,
            move_max_force=GRIPPER_MAX_FORCE,
            verbose=self.verbose,
            receive_latency=0.0
        )
        self.gripper.start(wait=True)
        print("[TeleopInterface] CustomGripperController started")

        self._running = True
        print("=" * 50)
        print("Teleop Interface Ready!")
        print("=" * 50)

    def stop(self):
        """停止所有控制器"""
        print("\n[TeleopInterface] Stopping...")
        self._running = False

        if self.robot is not None:
            try:
                self.robot.stop(wait=True)
                print("[TeleopInterface] Robot stopped")
            except Exception as e:
                print(f"[TeleopInterface] Error stopping robot: {e}")

        if self.gripper is not None:
            try:
                self.gripper.stop(wait=True)
                print("[TeleopInterface] Gripper stopped")
            except Exception as e:
                print(f"[TeleopInterface] Error stopping gripper: {e}")

        if self.shm_manager is not None:
            try:
                self.shm_manager.shutdown()
                print("[TeleopInterface] SharedMemoryManager shutdown")
            except Exception as e:
                print(f"[TeleopInterface] Error shutting down shm_manager: {e}")

        print("[TeleopInterface] Stopped")

    def is_ready(self):
        """检查所有控制器是否就绪"""
        robot_ready = self.robot is not None and self.robot.is_ready
        gripper_ready = self.gripper is not None and self.gripper.is_ready
        return robot_ready and gripper_ready

    def get_robot_state(self):
        """获取机器人状态"""
        if self.robot is not None:
            return self.robot.get_state()
        return None

    def get_gripper_state(self):
        """获取夹爪状态"""
        if self.gripper is not None:
            return self.gripper.get_state()
        return None

    def run(self):
        """主循环 - 保持运行直到收到中断信号"""
        self.start()

        # 设置信号处理
        def signal_handler(sig, frame):
            print("\nReceived interrupt signal")
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            print("\nTeleop Interface is running. Press Ctrl+C to stop.")
            print("Robot and gripper controllers are ready for commands.")

            # 打印当前状态
            while self._running:
                if self.is_ready():
                    robot_state = self.get_robot_state()
                    gripper_state = self.get_gripper_state()

                    if robot_state is not None:
                        pose = robot_state.get('ActualTCPPose', None)
                        if pose is not None:
                            print(f"\r[Robot] TCP Pose: xyz=({pose[0]:.3f}, {pose[1]:.3f}, {pose[2]:.3f})", end="")

                time.sleep(1.0)

        except KeyboardInterrupt:
            print("\nKeyboard interrupt")
        finally:
            self.stop()


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description="Teleop Interface (No ROS)")
    parser.add_argument("--robot-ip", type=str, default=FRANKA_IP, help="Franka NUC IP")
    parser.add_argument("--robot-port", type=int, default=FRANKA_PORT, help="Franka ZeroRPC port")
    parser.add_argument("--gripper-port", type=str, default=GRIPPER_PORT, help="Gripper serial port")
    parser.add_argument("--gripper-motor-id", type=int, default=GRIPPER_MOTOR_ID, help="Gripper motor ID")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    args = parser.parse_args()

    interface = TeleopInterface(
        robot_ip=args.robot_ip,
        robot_port=args.robot_port,
        gripper_port=args.gripper_port,
        gripper_motor_id=args.gripper_motor_id,
        verbose=args.verbose
    )
    interface.run()


if __name__ == "__main__":
    main()

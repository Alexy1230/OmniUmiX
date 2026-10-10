#!/usr/bin/env python3
"""
Franka 回零/检查姿态脚本。

说明:
1. 直接通过 ZeroRPC 调用中间层服务，不依赖本机安装 polymetis。
2. 读取的是服务端返回的原始 flange pose，不叠加额外 tip 或 45 度法兰补偿。
3. 默认目标不是关节全零，而是一组 Franka 常用的安全 home-like 姿态，
   方便现场观察末端法兰方向。
"""

import argparse
import time

import numpy as np
import zerorpc
from scipy.spatial.transform import Rotation


DEFAULT_HOME_Q = np.array(
    # [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.0],
    # [2.16, 1.20, -1.16, -1.83, -0.22, 1.76, 2.06],
    [0.83, 0.22, -0.06, -2.13, -0.56, 2.33, 1.99],
    dtype=float,
)


def parse_joint_list(text: str) -> np.ndarray:
    values = np.fromstring(text, sep=",", dtype=float)
    if values.size != 7:
        raise argparse.ArgumentTypeError(
            "Expected 7 comma-separated joint values, "
            f"but got {values.size}: {text!r}"
        )
    return values


def format_joint_report(joints: np.ndarray) -> str:
    joints_deg = np.degrees(joints)
    lines = []
    for idx, (rad, deg) in enumerate(zip(joints, joints_deg), start=1):
        lines.append(f"  q{idx}: {rad:+.6f} rad  ({deg:+7.2f} deg)")
    return "\n".join(lines)


def format_pose_report(pose: np.ndarray) -> str:
    pos = pose[:3]
    rotvec = pose[3:]
    rot = Rotation.from_rotvec(rotvec)
    quat_xyzw = rot.as_quat()
    euler_xyz_deg = rot.as_euler("xyz", degrees=True)

    lines = [
        f"  position [m]:      {np.round(pos, 6).tolist()}",
        f"  rotvec [rad]:      {np.round(rotvec, 6).tolist()}",
        f"  quat xyzw:         {np.round(quat_xyzw, 6).tolist()}",
        f"  euler xyz [deg]:   {np.round(euler_xyz_deg, 3).tolist()}",
    ]
    return "\n".join(lines)


def fetch_state(server):
    joints = np.asarray(server.get_joint_positions(), dtype=float)
    flange_pose = np.asarray(server.get_ee_pose(), dtype=float)
    return joints, flange_pose


def main():
    parser = argparse.ArgumentParser(
        description="Move Franka to a safe home-like inspection pose via ZeroRPC."
    )
    parser.add_argument(
        "--robot-ip",
        type=str,
        default="172.16.13.170",
        help="ZeroRPC server IP on the robot-side NUC",
    )
    parser.add_argument(
        "--robot-port",
        type=int,
        default=4242,
        help="ZeroRPC server port",
    )
    parser.add_argument(
        "--move-time",
        type=float,
        default=6.0,
        help="Joint move duration in seconds",
    )
    parser.add_argument(
        "--settle-time",
        type=float,
        default=1.0,
        help="Extra wait time after motion in seconds",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="ZeroRPC request timeout in seconds",
    )
    parser.add_argument(
        "--joints",
        type=parse_joint_list,
        default=None,
        help="Optional target joints as 7 comma-separated radians",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print current/target state only, do not move",
    )
    parser.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip Enter confirmation before motion",
    )
    args = parser.parse_args()

    target_q = DEFAULT_HOME_Q.copy() if args.joints is None else args.joints.copy()

    server = zerorpc.Client(timeout=args.timeout, heartbeat=20)
    server.connect(f"tcp://{args.robot_ip}:{args.robot_port}")

    try:
        curr_q, curr_pose = fetch_state(server)

        print("=" * 68)
        print("Franka Home / Flange Inspection")
        print("=" * 68)
        print(f"Endpoint: {args.robot_ip}:{args.robot_port}")
        print()
        print("Current joint positions:")
        print(format_joint_report(curr_q))
        print()
        print("Current raw flange pose:")
        print(format_pose_report(curr_pose))
        print()
        print("Target joint positions:")
        print(format_joint_report(target_q))
        print()
        print(
            "Note: this is a safe home-like inspection pose, not all-zero joints."
        )
        print("=" * 68)

        if args.dry_run:
            return

        if not args.no_confirm:
            input("Press Enter to move the robot to the target pose...")

        server.move_to_joint_positions(target_q.tolist(), float(args.move_time))
        print(
            f"Moving... waiting {args.move_time + args.settle_time:.1f}s for motion to finish."
        )
        time.sleep(args.move_time + args.settle_time)

        final_q, final_pose = fetch_state(server)
        joint_error = final_q - target_q

        print()
        print("Reached pose. Current joint positions:")
        print(format_joint_report(final_q))
        print()
        print("Joint error relative to target:")
        print(format_joint_report(joint_error))
        print()
        print("Current raw flange pose after move:")
        print(format_pose_report(final_pose))
        print("=" * 68)
    finally:
        server.close()


if __name__ == "__main__":
    main()

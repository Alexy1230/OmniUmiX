#!/usr/bin/env python3
"""
获取 Franka 机器人当前的末端执行器位姿
运行此脚本时，将机器人手动移动到你想要的初始位置，然后运行脚本获取位姿值
"""

import sys
import os

# 添加项目根目录到 Python 路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

import numpy as np
from umi.real_world.franka_interpolation_controller import FrankaInterface

def main():
    robot_ip = "172.16.13.65"  # 根据你的配置修改
    port = 4242
    
    print(f"连接到机器人 {robot_ip}:{port}...")
    robot = FrankaInterface(ip=robot_ip, port=port)
    
    # 获取当前末端执行器位姿
    ee_pose = robot.get_ee_pose()
    print("\n" + "="*60)
    print("当前末端执行器位姿 (End-Effector Pose)")
    print("="*60)
    print(f"位置 [x, y, z] (米):")
    print(f"  x = {ee_pose[0]:.6f}")
    print(f"  y = {ee_pose[1]:.6f}")
    print(f"  z = {ee_pose[2]:.6f}")
    print(f"\n旋转 [rx, ry, rz] (轴角表示):")
    print(f"  rx = {ee_pose[3]:.6f}")
    print(f"  ry = {ee_pose[4]:.6f}")
    print(f"  rz = {ee_pose[5]:.6f}")
    
    print("\n" + "-"*60)
    print("复制以下代码到 eval_real.py 中的 target_home_pose:")
    print("-"*60)
    print(f"""target_home_pose = np.array([
    {ee_pose[0]:.6f}, {ee_pose[1]:.6f}, {ee_pose[2]:.6f},  # x, y, z position
    {ee_pose[3]:.6f}, {ee_pose[4]:.6f}, {ee_pose[5]:.6f}   # axis-angle rotation
])""")
    print("-"*60)
    
    # 获取当前关节位置
    joint_positions = robot.get_joint_positions()
    print("\n当前关节位置 (Joint Positions):")
    for i, pos in enumerate(joint_positions):
        print(f"  关节 {i+1}: {pos:.6f} rad ({np.degrees(pos):.2f}°)")
    
    print("\n" + "-"*60)
    print("关节位置数组 (用于 move_to_joint_positions):")
    print("-"*60)
    print(f"initial_joint_positions = np.array({joint_positions.tolist()})")
    print("-"*60)
    
    robot.close()
    print("\n完成！")

if __name__ == "__main__":
    main()


# ssh 配置
ssh -N -f \
  -L 127.0.0.1:8099:127.0.0.1:8099 \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  vla-grasp.luoshaqi.baai-vision_vision.cn-beijing-daxing.ws@ssh.platform.baai.ac.cn

# 服务器连接
ssh -CAXY vla-grasp.luoshaqi.baai-vision_vision.cn-beijing-daxing.ws@ssh.platform.baai.ac.cn


# 机械臂nuc端启动
<!-- ./connect_nuc.sh -->
./connect_intel_nuc.sh
cd deoxys_control/deoxys/
./auto_scripts/auto_arm.sh config/charmander.yml

# 服务器启动
## UMI服务端
cd /share/project/lsq/universal_manipulation_interface && conda activate /share/project/lsq/miniconda3/envs/rdp
python policy_server.py

## RDP服务端
cd /share/project/tangll/workspace/Robot/reactive_diffusion_policy && conda activate /share/project/lsq/miniconda3/envs/rdp
python policy_server.py



# 客户端启动 (在/home/ps/reactive_diffusion_policy目录下)
## 快速启动
./quick_start_inference.sh

## 分别启动，开三个终端，分别跑 (rdp的conda环境) 
python camera_node_lanunch.py
python teleop_inference.py (这里运行之前要闭合夹爪，机械臂置为蓝灯)
python policy_client.py (如果要跑umi的客户端，运行python umi_policy_client.py)




# 注意事项
推理时一只手要放在机械臂急停器上，如果要发生碰撞按住急停器


rsync -avzP  -e ssh \
  /home/ych/Data-Scaling-Laws/data/checkpoints \
  vla-grasp.luoshaqi.baai-vision_vision.cn-beijing-daxing.ws@ssh.platform.baai.ac.cn:/share/project/lsq/universal_manipulation_interface/data/outputs/

  huggingface-cli download --resume-download --local-dir /share/project/lsq/universal_manipulation_interface/data/outputs/ \
Fanqi-Lin/Task-Models
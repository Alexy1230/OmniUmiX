1.nuc端
    终端1：
    ssh admin1@172.16.13.171 
    (password = 1234)

    conda activate polymetis

    launch_robot.py robot_client=franka_hardware

    终端2：
    ssh admin1@172.16.13.171 
    (password = 1234)

    conda activate polymetis
    cd Downloads/universal_manipulation_interface/scripts_real

    python ./launch_franka_interface_server.py

2.server端
    cd /share/project/lsq/Data-Scaling-Laws 
    conda activate /share/project/lsq/miniconda3/envs/rdp
    python policy_server.py


3.client端
    conda activate umi

    python umi_policy_client.py     --server-url http://127.0.0.1:8099     --robot-config example/eval_robots_config.yaml     --output data_local/eval_output --temporal-agg --init-joints

连接franka机械臂desk
google-chrome --incognito --new-window https://172.16.0.8

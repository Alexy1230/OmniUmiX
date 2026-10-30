# 独立客户端：各自支持单臂/双臂

两个独立入口，不使用 YAML 模式开关：

- `policy_client_pi_episodes.py`：纯视觉，单臂/双臂均支持。
- `policy_client_pi_proprio_episodes.py`：视觉+proprio，单臂/双臂请求均支持。

proprio 的 YAML 在原 `policy` 段内设置：

```yaml
proprio_history_seconds: 0.1
# 服务端已将动作转换为绝对值：
action_is_relative: false
```

不需要 `policy.proprioception`，由所运行的脚本决定模式；之前添加的该字段可以删除。

`robots`、`grippers` 和相机配置每臂一项：一项为单臂，两项为双臂；双臂 0=右、1=左。不需要另加 arm_order 配置。删除自定义 image_keys/state_key 即可使用自动键名。纯视觉模式保留原 action_is_relative 的语义，请沿用对应模型的设置。

| 模式 | 图像请求 | 实测状态历史 |
|---|---|---|
| 单臂纯视觉 | 当前一张 | 不发送 |
| 双臂纯视觉 | 当前右/左各一张 | 不发送 |
| 单臂 proprio | 当前/历史各一张 | `[2,10]` |
| 双臂 proprio | 右/左 × 当前/历史，共四张 | `[2,20]`，每行右10+左10 |

纯视觉仍发送当前 TCP 供服务端动作还原，但这不代表 TCP 输入网络。

## 运行

```bash
python policy_client_pi_proprio_episodes.py \
  --robot-config example/YOUR_CONFIG.yaml \
  --output data_local/policy_trial \
  --save-policy-data --debug-actions
```

检查后去掉 `--debug-actions` 执行。调试模式仍启动控制器并保留原夹爪置零行为。C 开始，S 结束，R 回初始关节，Q 退出。执行频率、轨迹筛选及碰撞处理沿用原 episode 客户端。

## Proprio 模型服务要求（不能混用 checkpoint）

**此前新增的训练适配器只支持单臂；本次修改的是客户端，不会把它变成双臂模型。双臂 proprio 在客户端侧已实现，但尚需训练端双臂历史适配器和对应 checkpoint，不能声称当前已可端到端运行。**

单臂图像键：`observation/image`、`observation/image_previous`。
双臂图像键：`observation/image_right`、`observation/image_left`、`observation/image_right_previous`、`observation/image_left_previous`。

其余键：`prompt`、`current_tcp_pose9_gripper`（10/20维）、`tcp_pose9_gripper_history`（2×10/20）、`episode_start_tcp_pose9_gripper`（10/20维）。当前参考与 history 最后一行相同。夹爪分别沿用每臂的标定，单位必须匹配训练数据。每个臂的绝对 TCP 均在它自身基座坐标系中。

服务端 metadata 必须声明 `proprioception=true`、`image_history_horizon=2`、`is_relative=false`，以及 `state_dim=32`（单臂）或 `64`（双臂，每臂32维本体状态）。双臂另需 `num_robots=2`、`arm_order=['right','left']`。客户端会在启动硬件前拒绝明显错配；这些声明必须来自真实实现，不要只修改 metadata 来绕过检查。

## 时间与 episode

使用环境时间轴请求 `[t-gap,t]`，而不是上一次推理。相机取最近真实帧，TCP/夹爪插值到目标时间；实际同步仍依赖传感器延迟配置和采样质量。

每轮第一个模型请求捕获固定起始位姿。历史早于该时刻时，同时填充各臂首帧图像和状态。C 开始下一轮时重新初始化。`--save-policy-data` 额外保存完整 `*_proprio_input.npz`。

## 文件及验证

修改 proprio 入口、`umi/real_world/proprio_policy_client.py`、`tests/test_proprio_client.py` 和本说明。未修改训练代码或原视觉入口。无硬件测试覆盖单/双臂请求、episode 重置和模型臂数错配；未进行实际机器人联调。

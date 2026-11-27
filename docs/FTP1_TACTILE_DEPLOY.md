# 新基线上的 sort_diff 触觉部署

- 正确基线：`/media/xcr/Data/LXY/new_deployment/omni-client-lxy`。
- 本次新目录：`/media/xcr/Data/LXY/omni-client-lxy-force`。
- 原基线与旧 `/media/xcr/Data/LXY/omni-client-force` 均不修改；不复制历史录像、日志和缓存。
- 新基线的动作执行循环、夹爪逐帧执行、时延参数、TCP变换、相机处理及两次模型预热保留。

## 运行顺序

1. 启动原有 sort_diff server，客户端连接后输入 `calibrate` / `CALIBRATE` 并回车。空行或拼写错误只会重新提示，不会自动取消；只有 `q/quit/cancel` 明确取消。
2. 此时不要夹物体，也不要让两块触觉垫互相挤压；不需要手动把夹爪闭合。
3. 确认后做触觉无接触校准，导出本次runtime，建立Solver。运行期间不在每次C/R时重标定。
4. 启动相机后，夹爪以低力矩向闭合方向机械回零，确认停转后才设置电机零点。
5. 再启动机械臂控制器；按原基线的配置归位，夹爪打开到 YAML `initial_position`。
6. 连续三个新反馈确认夹爪到位后，进行两次真实观测预热；返回动作全部丢弃。
7. 出现 Ready 后等 C 开始任务；S结束，R回初始关节并恢复夹爪开口，Q退出。

触觉校准和电机归零是两个独立步骤。不要先让物体夹在夹爪里，再把接触状态校准成零参考。
机器人/夹爪在初始化阶段会按上述要求动作，只有模型预测动作会等C才执行。
R只恢复初始位姿/开口，不重新设置电机或触觉零点。

## 夹爪机械归零

参考旧代码 `openpi_infer/omniumix/real_world/gripper_lk/controller.py:start()`，
沿用小负力矩和低速度连续停转判断。没有复制旧代码的角度/宽度倍数换算；仍使用新基线的 width_to_rad。
区别是归零失败会报错停止，不会像旧实现那样超时后仍把任意当前位置置零。
另外必须先观察到至少0.02rad的实际闭合行程再接受低速停转；完全没动不会当成机械归零成功。
若启动时已经完全闭合，或扭矩不足导致完全不动，会停止并保留零点，需要无负载稍微张开后再试。

```yaml
grippers:
  - gripper_type: custom
    homing:
      enabled: true
      torque_nm: -0.05
      timeout_s: 2.0
      velocity_threshold_rad_s: 0.05
      stall_samples: 10
      min_duration_s: 0.2
      min_travel_rad: 0.02
    initial_position: 0.055
```

这些 homing 默认值来自旧归零方法，其中最短检测时间用于避免初始静止被立即误判为限位。
若报归零超时，应检查机械方向、阻挡、反馈及超时时间，不应改成“超时照样归零”。
必须无物体/外部阻挡；停转本身不能区分机械限位与夹住物体。
RAM零点对应 width_to_rad.min_rad=0；回零后校验角度接近0。
`initial_position` 是标定宽度（米），不是把负 driver_width 取绝对值。
本次 sort 模板保留你在旧 force 模板明确填过的标定及0.055初始开口，不再用通用示例覆盖。

## 配置与运行

使用 `example/eval_ftp1_sort_diff.yaml`，检查 server_url、机器人端点/TCP、MVS序列号、夹爪串口。
该专用模板继承新基线硬件配置，保留原 sort 模板已填写的两指 OG000039/OG000052 与夹爪标定/初始开口。
序列号映射来自已有配置，不是自动识别物理左右；换设备需重新确认。

```bash
cd /media/xcr/Data/LXY/omni-client-lxy-force
conda activate omniumi
python policy_client_pi_wolatency.py -rc example/eval_ftp1_sort_diff.yaml -o data_local/sort_diff_seq
```

也可用 `policy_client_pi_episodes.py`、`policy_client_pi_episodes_new.py`、
`policy_client_pi.py`、gripper_tuned及proprio入口，各自原有执行参数保持；查看对应 `--help`。
新基线的 episodes.py 为 naive 逻辑、episodes_new.py 为另一种时延调度，二者没有合并或互换。
`zero_action_client_pi.py` 保持实测位姿，明确跳过机械归零/初始开口移动，避免 dry-run 变成闭合动作；
它要求已有正确电机零点，且仍启动保持控制器。

## Difference PNG 的保存位置

**直接运行主client就会保存，不需要先运行或同时运行单独的保存脚本。**
默认 `policy.save_tactile_images: true`，每次模型推理都无损保存，包括两次预热。
终端会打印 `[Capture] Enabled for this CLIENT run`，以及首个 warmup/policy 图像的绝对目录。
这里保存的是实际推理请求的图像序列，不是固定30Hz全程触觉视频。

```text
<output>/policy_inputs/run_*/request_000001/
  left_difference_rgb.png             # 原尺寸700x400
  right_difference_rgb.png
  tactile_left_difference_rgb.png     # 实际请求图像224x224
  tactile_right_difference_rgb.png
  image_rgb.png                      # 场景图像
  state.npy                          # 实际请求pose10，夹爪为模型弧度
  metadata.json                      # phase、时间戳、runtime及哈希、夹爪标定
```

文件名中的request编号是推理请求序号，不是训练episode的frame_index。
metadata.phase=warmup表示预热；phase=policy表示正式推理。
响应和动作保存在 `<output>/policy_responses.jsonl`，input_directory关联到对应图像目录。
PNG写入由单个后台线程处理，队列有上限；写入失败会报错，不会静默漏帧。
磁盘跟不上时会等待写入；需要纯速度测试可在policy中设 `save_tactile_images: false`。

与训练对比时，应选择训练数据的 `observation.images.xense_right_hand_left_finger_difference` / `right_hand_right_finger_difference`，不要拿它和原始 Rectify 视频比较。
优先查看实际请求的224图像；用原尺寸Difference进一步检查参考、颜色和接触形变。
训练视频有H.264/yuv420p有损压缩，PNG无损，因此不承诺压缩后像素逐字节相同。

## 可选：不连接机器人的设备诊断（不是主client保存的前置步骤）

```bash
python scripts/test_xense_camera.py --config example/eval_ftp1_sort_diff.yaml \
  --seconds 10 --output data_local/tactile_check --show
```

它只启动触觉进程，不导入机器人环境，不执行电机机械归零。
输出 `run_*/frame_*/left_difference_rgb.png`、right及model224图像、时间戳/runtime信息。

## Server 和触觉处理

server不用改：原三字段握手兼容，输入为场景RGB、当前TCP/夹爪参考、prompt及双指Difference；
sort_diff输出仍为actions[32,10]，已经是绝对pose10，不依赖力输入/力预测。
触觉仍在 `omniumix_lerobot_v21` 环境独立运行，使用实际采集的SDK/ONNX Runtime。
处理依据 `/home/xcr/pbg/omniumix_lerobot`：
Rectify BGR -> 导出runtime对应的createSolver -> Solver Difference -> RGB -> INTER_LINEAR 224。
CPU/协议测试和真实录制帧的处理对照不等于实机抓取成功验证；本次不执行真实机器人或夹爪测试。

## 主要文件

- bimanual_policy.py / proprio_policy_client.py：共享FTP1触觉协议及输入保存。
- xense_capture.py：独立触觉采集、校准和runtime/Solver；沿用已核对的实际采集流程。
- gripper_homing.py / gripper_controller_1_14.py：只新增受控机械回零，不改安装的motor包。
- bimanual_umi_env.py：先机械归零再启动臂控制器；启动失败清理已启动组件。
- client_deployment.py：保留原预热/时延工具，增加初始开口反馈确认及预热标记。
- policy_input_recorder.py：后台无损输入保存。
- 原各client：仅接入配置/日志和反馈等待，执行循环保留。


### 夹爪不动时看哪些输出

`calibrate` 首先处理触觉，只有 Xense 和 MVS 就绪后才出现 `[Gripper home] BEGIN`。
出现 BEGIN 后每0.25秒打印 angle、vel、requested_tau、measured_tau、closing_travel。
没有BEGIN说明还未进入机械归零，不是调大力矩可以解决的问题。
若实际闭合行程始终为0，不会重置电机零点；反馈有力矩但不动可能是已闭合或静摩擦/阻挡，
反馈无力矩也需检查指令是否生效。当前归零扭矩仍为-0.05Nm，没有自动增加。
`homing.torque_nm` 控制归零；`force_limit_nm` 控制后续位置命令的力矩上限，二者不同。

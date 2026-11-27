# UMI 风格本体状态输入（单臂 pi0.5）

新增训练配置：`pi05_umi_lift_bottle_proprio`。原纯视觉配置、双臂配置保持原行为。
本次只修改训练/模型服务仓库，没有修改机器人客户端。需要重新微调；不能给旧纯视觉 checkpoint 加字段就获得本体状态能力。

## 输入是什么

使用两帧 RGB 图像、任务文本和两帧实测 TCP/夹爪状态；仅 proprio 配置启用历史图像。沿用本仓库的列式 rotation-6D：

```text
原始单帧 observation.state（10 维）：
[x, y, z, R[:,0] 的3个数, R[:,1] 的3个数, gripper]
```

每个 episode 内，原始位姿必须位于同一个固定坐标系，位置单位为米。夹爪保持数据集原单位（角度还是宽度必须由实际数据确认），代码不猜测或修改单位，客户端转换后的夹爪值必须与之相同。

默认取 `[t-3, t]` 两帧；30 Hz 数据对应相隔 0.1 秒。episode 开头不足三帧时，旧帧取本 episode 第一帧，不跨 episode。图像和状态共用完全相同的两个数据行索引；通过 LeRobot 解码对应行的历史图像。episode 第一帧复用当前图像，避免重复解码。每个普通样本多一次历史行读取/视频解码（该读取也会触发 LeRobot 的动作窗口查询），可能增加 I/O 开销。

每帧构造 16 维，按旧帧→最新帧拼成 32 维：

| 内容 | 维数 | 计算 |
|---|---:|---|
| 相对最新观测的 TCP 位置 | 3 | `inv(T_t) @ T_i` 的平移 |
| 相对最新观测的 TCP 旋转 | 6 | 同一变换的列式 rotation-6D |
| 夹爪开度/角度 | 1 | 原始绝对值，不作相减 |
| 相对 episode 起点的旋转 | 6 | `inv(T_start) @ T_i` 的列式 rotation-6D |

最新帧的前 9 维是零平移和单位旋转；并不是 9 个零。历史帧、夹爪和起点相对旋转仍携带信息。
同一 episode 整体换世界坐标系，上述相对状态不变；episode 内的跟踪漂移、重定位跳变和 TCP 标定错误不会自动消失。

这复用了官方 UMI 的本体状态含义，但不是复制其 Diffusion Policy 网络。pi0.5 使用已有的离散 state token 输入：先对 32 维状态做分位数归一化，再用 `discrete_state_input=True` 编码到文本 token 中；无需新增神经网络层。

与官方 UMI 的明确差别：本配置使用两帧图像、32 步动作；起点取**当前数据集 episode 的第一条 observation.state**，不读取官方的 `demo_start_pose`，也不加入起点随机噪声。若数据经过裁剪，该起点不是原始采集的最开始一帧。部署开始状态应覆盖训练时的起点分布。

## 动作表示

训练将绝对 `action[32,10]` 转为相对最新实测 `observation.state` 的动作：

```text
T_relative[k] = inv(T_observation_latest) @ T_action[k]
```

不再强制第一条 action 作为参考。若第一条标签恰好等于当前观测，它仍然是单位运动，这是数据的结果。
夹爪保持绝对值；xyz 和夹爪沿用原 UMI action 归一化，动作 rotation-6D 不归一化。

模型内部动作仍是 `[32,32]`（后面补零），服务输出仍是 `[32,10]` 的绝对 pose9 + gripper，使用本次请求的最新 TCP 还原。已有输出转换类名字含 `FromFirstAction`，但其反变换实际使用传入的 `current_tcp_pose9_gripper`，适用于此处。

## 历史图像如何进入网络

官方 OpenPI 支持多个图像输入槽，LeRobot 支持时间偏移查询，但该 pi0.5 流程没有直接开启图像时间历史的配置开关。本次复用现有图像槽，不给图像强加 `[B,T,H,W,C]` 维度，不修改模型主体或权重形状：

| 模型内部图像槽 | proprio 配置中的含义 | mask |
|---|---|---|
| `base_0_rgb` | 全零占位 | False |
| `left_wrist_0_rgb` | 同一相机的历史图像 `t-3` | True |
| `right_wrist_0_rgb` | 同一相机的当前图像 `t` | True |

这里的 left/right 只是模型槽名，不表示左右手。固定槽位区分时间角色，两帧都走相同的 wrist 图像预处理路径。图像沿用适配器的 RGB/uint8 转换、缩放至 224×224 和模型原有训练增强；与以前使用 base 槽不同，两帧不会走 base 槽独有的随机几何裁剪/旋转。没有新增时间嵌入或循环记忆，也不是完整复制 UMI 的网络结构。

纯视觉配置仍是 `base_0_rgb` 当前图像有效、两个 wrist 槽屏蔽。不要直接恢复之前只使用当前图像的 proprio 训练实验；新的实验名应区分双帧版本。状态/动作数学不变，已有正确的 proprio 统计量可以复用，但不能使用纯视觉统计量。

## 数据要求

LeRobot 数据必须提供：

- `observation.images.fisheye_img`：当前 RGB 图像；沿用现有图像预处理。
- `observation.state`：真实观测的绝对 `[10]`，不是关节角、不是 7/8 维 pose，也不能用未来 action 代填。
- `action`：绝对 `[10]` 单帧标签；加载器生成 32 帧 chunk。
- `episode_index`、从 0 连续递增的 `frame_index`，每个 episode 的行连续排列。
- 原有任务文本来源：LeRobot task 元数据，或配置的 `default_prompt`。

当前没有访问服务器数据，因此无法确认你的实际 parquet 是否全部符合这些条件。维度/旋转/episode 边界不满足时会报错，不会静默猜测转换。

## 同步到服务器的文件

修改：

1. `src/openpi/training/config.py`：新数据配置、新训练配置，启用 pi0.5 state token。
2. `src/openpi/training/data_loader.py`：按 episode 提取状态历史，接入专用归一化。
3. `src/openpi/policies/policy_config.py`：推理服务使用相同的状态归一化；动作输出接口不变。

新增：

4. `src/openpi/policies/umi_proprio_policy.py`：成对图像/状态历史采样、图像槽适配、32 维状态、观测参考动作转换和校验。
5. `src/openpi/policies/umi_proprio_policy_test.py`：几何、边界、训练/推理一致性与 state token 传递测试。
6. 本说明。

## 服务器使用

在已经可以训练的完整 OpenPI 仓库/环境中同步上述文件。本地目录是代码副本，没有完整环境依赖清单。

先在 `config.py` 新的 `pi05_umi_lift_bottle_proprio` 条目中确认 `repo_id`。默认沿用原 lift bottle 数据路径；其他任务可复制该配置条目，修改名称、数据路径和独立 `asset_id`，保留 `proprioception=True`。

```bash
# 在服务器完整仓库根目录，激活原训练环境后执行
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest src/openpi/policies/umi_proprio_policy_test.py -q

# 必须重新统计：新状态和新的动作参考都需要新的统计量
python scripts/compute_norm_stats.py --config-name pi05_umi_lift_bottle_proprio

# 使用新实验名，从配置指定的 pi05_base 权重开始微调
python scripts/train.py pi05_umi_lift_bottle_proprio --exp-name lift_bottle_proprio_history_v1
```

不要复用旧 `umi_lift_bottle` 的 norm_stats，也不要 resume 旧纯视觉实验。新 asset_id 是 `umi_lift_bottle_proprio`。训练步数、batch size、LoRA 设置沿用原配置；不是保证加入本体状态就能消除执行误差。

## 部署客户端需要补什么

新模型服务请求：

```python
{
    "observation/image": rgb_image,                # 当前 HWC uint8 RGB
    "observation/image_previous": previous_image, # 历史 HWC uint8 RGB，与 history[0] 对应
    "prompt": "lift the bottle",
    "tcp_pose9_gripper_history": history,         # float32 [2,10]，旧帧→当前帧
    "episode_start_tcp_pose9_gripper": start,     # float32 [10]，每次 episode 开始保存一次
    # 可选；如提供必须等于 history[-1]：
    "current_tcp_pose9_gripper": history[-1],
}
```

这些位姿在部署时用机器人基座坐标系即可，但必须使用训练对应的 TCP 定义、旋转排列和夹爪单位。两帧历史应按**时间间隔**对齐（默认 0.1 秒），不能用“前一次网络请求”代替；实际客户端采样率不一定等于数据集帧率。两帧状态分别要与两帧图像的采集时间对齐。代码只保证训练数据行索引一致，不会修复采集时已存在的时间同步误差。部署客户端必须维护图像/状态缓冲并按时间匹配，不能拿上一次推理图像替代固定时间间隔的历史图像。

每次新 episode 重置起点和历史，起点只在 episode 开始保存一次。不要每次请求重新设起点，否则起点相对旋转信息会消失。服务端自动构造/归一化 32 维 state；客户端不要提前归一化。

旧客户端缺少新增字段（包括历史图像）时会报错，这是为避免无声退化为纯视觉推理。必须补齐上述输入后才部署新模型。

## 已验证范围

本机通过语法检查及 14 项隔离的数学/适配器检查（坐标系不变性、动作往返、历史边界、图像/状态采样索引一致性、边界填充、历史缺失报错、训练/推理一致性、归一化与状态传入 tokenizer 等）。本机缺少 JAX/Flax/LeRobot 完整依赖，隔离检查替代了依赖外壳；尚未运行完整 pytest 集成、真实数据加载、GPU 训练或机器人部署。上面的服务器测试命令用于在真实环境复核。

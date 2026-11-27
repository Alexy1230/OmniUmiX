# FastTWAM 触觉图像使用方式与历史训练记录

整理日期：2026-10-28。

本文整理 2026 年 9 月 FastTWAM 实验中触觉图像的处理、模型融合、历史训练版本、深度监督和在线接入方式。记录依据包括原始训练配置、保存的 checkpoint、数据处理代码和接入笔记。文中的 checkpoint 进度为本次核查时所见的保存记录。

主要资料位于：

- FastWAM 工程：`/share/project/dcw/projects/FastWAM`。
- OmniUmiX 工程：`/share/project/lxy/OmniUmiX_Dataset_DCW`。
- [FastTWAM 接入记录](/share/project/lxy/OmniUmiX_Dataset_DCW/docs/debug/twam/README.md:1)。

## 1. 整体方案

之前的核心做法是：先把左右两路触觉图像编码成特征，再将这些特征作为条件 token 输入 FastWAM。后续版本增加了四帧触觉历史，以及使用未来触觉指导训练的教师学生蒸馏。

这套流程包含两个不同的编码模块：

| 模块 | 输入与输出 | 策略训练阶段是否更新 |
| --- | --- | --- |
| ResNet18 触觉图像编码器 | 每侧单帧触觉 RGB 图像 → 512 维特征 | 固定；特征提前离线计算 |
| `TactileFeatureContextEncoder` | 512 维特征 → 4096 维条件 token | 参与策略训练 |

## 2. 触觉图像怎样变成训练输入

### 2.1 单帧图像预处理与编码

每侧触觉图像使用相同的处理链路：

```text
Xense rectify RGB 图像，700×400×3
    ↓ 像素除以 255，双线性抗锯齿缩放
RGB tensor，3×320×240
    ↓ 预训练 ResNet18 编码器
每侧 512 维 float32 特征
    ↓ 左侧在前、右侧在后
每个时刻共 1024 维触觉特征
```

左右传感器共用同一个图像编码器。虽然目录名字叫 `TacVAE v2`，实际输出是确定性的 ResNet18 特征，没有 VAE 随机采样。

图像预处理将 RGB 像素缩放到 `[0,1]`，并将图像缩放到高 320、宽 240。在线编码时保留原始 700×400 图像，由编码器执行缩放，以保持与离线特征提取流程一致。

实现依据：[触觉图像编码器](/share/project/lxy/OmniUmiX_Dataset_DCW/omniumix/real_world/tac_xense_vae/encoder.py:49)。

### 2.2 离线特征提取与数据集写入

训练前会离线提取每张触觉图像的特征，先保存成左右两个 NPZ，再根据时间戳最近邻匹配到数据集的主采样帧。

```text
原始 episode：左右 Xense rectify 图像及时间戳
    ↓ 分别编码每侧每一帧
tacvae_xense_v2_left.npz
tacvae_xense_v2_right.npz
    ↓ 按时间戳最近邻匹配到主采样帧
LeRobot 数据集中的左右触觉特征字段
```

写入的数据字段为：

```text
observation.tac.xense.vae.left         [512]
observation.tac.xense.vae.right        [512]
observation.tac.xense.vae.left_valid
observation.tac.xense.vae.right_valid
```

因此，策略训练读取的是已经算好的触觉特征，ResNet18 图像编码器在策略训练阶段保持固定。

离线处理与时间对齐规则见：[触觉数据集说明](/share/project/lxy/OmniUmiX_Dataset_DCW/docs/sensors/tac_xense_vae/README.md:144)。

## 3. FastWAM 怎样利用触觉特征

### 3.1 将特征转换为条件 token

`TactileFeatureContextEncoder` 与前面的 ResNet18 图像编码器是两个模块。对每一侧、每一个时刻的 512 维特征，其处理过程为：

```text
512 维特征
    ↓ LayerNorm
Linear(512,1024) → SiLU → Linear(1024,4096)
    ↓ 加上传感器身份 embedding、时间位置 embedding
LayerNorm
    ↓
4096 维条件 token
```

传感器身份 embedding 区分左右两侧；时间位置 embedding 区分历史中的各个时刻。

| 触觉输入 | 条件 token 数量 | token 形状，不含 batch 维 |
| --- | --- | --- |
| 当前一帧、左右两侧 | 2 | `[2,4096]` |
| 四帧历史、左右两侧 | 8 | `[8,4096]` |

实现依据：[触觉条件编码器](/share/project/dcw/projects/FastWAM/src/fastwam/models/wan22/tactile_context_encoder.py:36)。

### 3.2 条件融合位置

触觉 token 会拼接到语言条件序列后面，供模型读取。

- 早期单帧版本：包含触觉的条件同时用于 VideoDiT 和 ActionDiT。
- 后续 CFD-V1.1 版本：分开视频与动作两套条件，触觉条件直接加入 ActionDiT 分支。

对应实现见：[条件拼接](/share/project/dcw/projects/FastWAM/src/fastwam/models/wan22/fastwam.py:333)、[视频与动作条件分配](/share/project/dcw/projects/FastWAM/src/fastwam/models/wan22/fastwam.py:498)、[CFD-V1.1 分支设置](/share/project/dcw/projects/FastWAM/src/fastwam/models/wan22/fastwam_v0.py:93)。

### 3.3 触觉特征的归一化

触觉特征进入策略前，数据处理器主要做 float32 转换和左右拼接，没有套用动作的分位数归一化；条件编码器内部使用 LayerNorm 处理这些特征。

需要区分两处处理：

- 图像编码前：RGB 像素除以 255，得到 `[0,1]` 图像输入。
- 策略条件编码时：对 512 维特征使用 LayerNorm，再投影为条件 token。

数据处理依据：[FastWAM 数据处理器](/share/project/dcw/projects/FastWAM/src/fastwam/datasets/lerobot/processors/fastwam_processor.py:293)。

## 4. 历史训练版本与保存记录

### 4.1 版本区别

| 版本及日期 | 策略使用的触觉 | 未来触觉的作用 |
| --- | --- | --- |
| PatternClass 单帧版，2026-09-07 | 当前一帧，左右各 512 维 | 未使用 |
| CFD-V1.1 Teacher，2026-09-11 至 12 日 | 4 帧历史，另外加入 32 步未来触觉 | 为教师提供训练时的额外信息 |
| CFD-V1.1 Student，同一批实验 | 学生分支使用 4 帧历史 | 冻结教师读取未来触觉，指导学生训练 |
| 无蒸馏对照，2026-09-20 | 4 帧历史 | 关闭未来触觉和蒸馏 |

无蒸馏对照的一个配置依据是：[PegInHole 无蒸馏实验配置](/share/project/dcw/projects/FastWAM/runs/umi_peg_in_hole_thick_thin_tacvae_jointaction10_nocfd_fps30/peg_in_hole_jointaction10_nocfd_2026-09-20_00-52-33/config.yaml:85)。

### 4.2 PatternClass 单帧版

训练目录：

```text
/share/project/dcw/projects/FastWAM/runs/
  umi_pattern_class_tacvae_singleframe_fps30/
    2026-09-07_20-36-42/
```

关键记录：

- `tactile_context.enabled: true`。
- 每侧特征维度为 512，传感器数量为 2。
- `history_steps: 1`，使用单帧触觉。
- 已保存 checkpoint 到 `step_014060.pt`。
- 2026-09-08 接入笔记里写的“只到 5000 步”是当时的中间进度。

原始配置：[PatternClass 单帧训练配置](/share/project/dcw/projects/FastWAM/runs/umi_pattern_class_tacvae_singleframe_fps30/2026-09-07_20-36-42/config.yaml:84)。

### 4.3 AdjustSlipTube 教师学生版本

教师训练目录：

```text
/share/project/dcw/projects/FastWAM/runs/
  umi_adjust_slip_tube_thick_thin_tacvae_cfdv11_jointaction10_teacher_fps30/
    adjust_slip_tube_jointaction10_teacher_2026-09-12_00-41-03/
```

学生训练目录：

```text
/share/project/dcw/projects/FastWAM/runs/
  umi_adjust_slip_tube_thick_thin_tacvae_cfdv11_jointaction10_student_fps30/
    adjust_slip_tube_jointaction10_student_2026-09-12_00-41-03/
```

关键记录：

- 教师和学生都保存到 `step_015640.pt`。
- 学生配置明确引用教师的 `step_015640.pt`。
- 教师使用 4 帧历史触觉和 32 步未来触觉。
- 学生分支使用 4 帧历史触觉。
- 模型预测完整 10 维动作。
- 学生使用 `lambda_flow=0.1` 的教师 flow 蒸馏，`lambda_token=0.0`。

学生配置里仍然包含未来触觉字段，是因为训练时需要运行冻结的教师分支。部署学生时不需要未来触觉。

依据：[学生原始配置](/share/project/dcw/projects/FastWAM/runs/umi_adjust_slip_tube_thick_thin_tacvae_cfdv11_jointaction10_student_fps30/adjust_slip_tube_jointaction10_student_2026-09-12_00-41-03/config.yaml:98)、[试管实验记录](/share/project/dcw/projects/FastWAM/docs/真机OmniWAM_AdjustSlipTube_CFD-V1.1_JointAction10方案与实现记录.md:41)、[教师与学生条件构建代码](/share/project/dcw/projects/FastWAM/src/fastwam/models/wan22/fastwam_v0.py:315)。

## 5. 深度信息当时用在哪里

深度用于预训练触觉图像编码器。原始配置中，ResNet18 从单帧触觉 RGB 提取 512 维特征，重建头同时学习重建 RGB 和深度，损失权重为：

$$
L_{\mathrm{encoder}} = L_{\mathrm{RGB}} + 0.5 L_{\mathrm{depth}}
$$

依据：[触觉编码器预训练配置](/share/project/dcw/projects/OmniUmiX/TacVAE/univtac/configs/real_omniwam_resnet18_marked_rgb_depth.yaml:80)。

提取策略训练特征时只运行编码器，不运行深度 decoder，也不将重建深度图单独送入策略。因此，深度的作用是通过重建监督训练触觉表示，FastTWAM 实际消费的是这个表示。

## 6. 在线执行时怎样接入

最早的 FastTWAM 客户端每次请求推理前执行以下流程：

```text
取左右最新一帧触觉 RGB 图像
    ↓ 同一份固定的 ResNet18 编码器分别编码
左侧 [512]，右侧 [512]
    ↓ 拼接
tactile_proxy: [1024]，float32
    ↓ 随鱼眼图像等信息发送 POST /dp
FastWAM server 将触觉特征传入模型
    ↓
生成动作序列
```

客户端实现：[最新触觉帧编码与特征拼接](/share/project/lxy/OmniUmiX_Dataset_DCW/scripts_infer/twam/policy_client_twam.py:298)。

这份客户端对应早期单帧模型。后续四帧模型的训练配置和代码也有保存，但这里找到的单帧客户端实现，不能直接证明四帧模型当时已经完成真机部署。

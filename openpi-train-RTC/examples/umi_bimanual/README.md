# Bimanual UMI Pi0.5 training and deployment

This adds `pi05_umi_bimanual` alongside the existing single-arm UMI configs.
It preserves their image-and-language conditioning and first-action-relative
training convention. It does not add numerical proprioception or inter-gripper
features. All action/reference blocks are **right first, left second**.

## Files changed

New files:

- `src/openpi/policies/umi_bimanual_policy.py`: `UmiBimanualRepack`,
  `UmiBimanualInputs`, `UmiBimanualOutputs`, and `make_umi_bimanual_example`.
  Reuses the single-arm `_parse_image` helper. Both named wrist slots are active;
  the base-image slot is zero and masked out. Semantic state is empty.
- `src/openpi/transforms_umi_bimanual.py`: independent right/left relative and
  absolute transforms, reusing single-arm pose-matrix helpers. Normalizes both
  xyz/gripper blocks while leaving column-wise rotation6d and padding unchanged.
- Adjacent policy/transform tests and `src/openpi/training/umi_bimanual_config_test.py`.

Modified files:

- `src/openpi/training/config.py`: `LeRobotUMIBimanualDataConfig`,
  `_make_umi_bimanual_train_config`, `pi05_umi_bimanual`, and `umi_action20` mode.
  The factory reuses `_make_umi_train_config` for Pi0.5 LoRA and optimizer settings.
- `src/openpi/training/data_loader.py`: selects 20-D normalization and the same
  half-frame video timestamp tolerance already used for single-arm UMI.
- `src/openpi/policies/policy_config.py`: selects matching normalization and
  unnormalization when serving a bimanual checkpoint.

Single-arm adapters, pose helpers, configurations, and their normalization remain
unchanged. `Policy.infer` already preserves `current_tcp_pose9_gripper` separately
from model inputs, so it needs no changes for a 20-D reference. The existing
compute-norm-stats script already runs data transforms first and skips empty state.

## 1. Prepare a synchronized LeRobot dataset

The local data converter is still single-arm; this change does not turn two
separate datasets into a synchronized bimanual dataset. Supply one dataset with
both wrist images and one combined action at each shared timestamp.

Default feature names (chosen interface; not verified against your future dataset):

| Field | Per-frame format |
|---|---|
| `observation.images.fisheye_img_right` | right wrist RGB image |
| `observation.images.fisheye_img_left` | left wrist RGB image |
| `action` | float vector of length 20, absolute poses |
| `task_index` / LeRobot task metadata | instruction used as prompt |

`observation.state`, if present, is ignored. The dataloader assembles `action`
into a 32-step chunk. Use the configured `right_image_key` and `left_image_key`
fields of `LeRobotUMIBimanualDataConfig` if your dataset uses different names.
Do not reverse their physical assignment.

Action values:

```text
[right_xyz(3), right_rot6d(6), right_gripper(1),
 left_xyz(3),  left_rot6d(6),  left_gripper(1)]
```

XYZ is in metres. Rotation6d is `[R[:,0], R[:,1]]` flattened column-wise, not
rows and not interleaved columns. Gripper values are absolute scalars in the
same units used by the dataset; the adapter does not convert mm/metres/radians.
The client scaling must match those units. For the current 0.03 m/rad driver,
client scales 1000/30 and .03 assume model gripper values are motor radians.
Confirm this from your dataset before training/deployment.

Both image streams should contain the same crop, orientation, exposure conventions
as inference. 240x240 RGB is the intended source format; the existing model
transforms resize/pad images to 224x224. Float CHW LeRobot images in [0,1] and
uint8 HWC client images are accepted. One current image per wrist is used, not
the two-frame history used by official UMI Diffusion Policy.

Training converts each arm's 32-step pose chunk to that arm's first-action frame:

```text
right_relative[k] = inverse(right_absolute[0]) @ right_absolute[k]
left_relative[k]  = inverse(left_absolute[0])  @ left_absolute[k]
```

No base-to-base transform is required for this image-only conditioning design.
It does not explicitly teach inter-hand pose features. Each arm's first pose is
identity after conversion; both gripper sequences retain their absolute values.

## 2. Set dataset and prompt in the config

In `src/openpi/training/config.py`, find the `_CONFIGS` entry:

```python
_make_umi_bimanual_train_config(
    name="pi05_umi_bimanual",
    repo_id="REPLACE_WITH_BIMANUAL_LEROBOT_DATASET",
    asset_id="umi_bimanual",
)
```

Replace `repo_id` with the real dataset path/repo ID. The placeholder cannot
train. By default prompts come from LeRobot task metadata, as in single-arm UMI.
For a fixed instruction, add `default_prompt="your actual task instruction"`.
Use the same instruction when deploying. The image-key fields are declared on
`LeRobotUMIBimanualDataConfig`; change their defaults if needed for this dataset.

Settings inherited from single-arm training:

- Pi0.5, LoRA vision/language and action expert variants, base checkpoint loader.
- `action_horizon=32`; internal `action_dim=32`, with 20 semantic dimensions and
  12 padding dimensions. Do not change internal dimension to 20 when reusing
  the configured pretrained weights.
- `discrete_state_input=False`; empty measured model state.
- Batch size 32, 20,000 training steps, existing optimizer/freezing settings.
- Metadata fps is 30. Dataloader action spacing uses dataset metadata fps;
  update the new config's metadata and client `control_fps` together if yours
  differs. The client's example currently says 20 Hz: verify and align it.

## 3. Compute fresh statistics, test, train

Use your working OpenPI training environment from this repository root. This
supplied tree does not contain a pyproject.toml/uv.lock, so use its existing
installed dependencies rather than assuming `uv sync` can initialize it here.
Expose the local source and client package so imports use these edited files:

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
python -m pytest -q src/openpi/transforms_umi_test.py \
  src/openpi/policies/umi_policy_test.py \
  src/openpi/transforms_umi_bimanual_test.py \
  src/openpi/policies/umi_bimanual_policy_test.py \
  src/openpi/training/umi_bimanual_config_test.py



export WANDB_MODE=disabled or
[.venv/bin/wandb login --relogin
export WANDB_ENTITY= ... ]
export OPENPI_DATA_HOME=/share/project/lxy/openpi/.openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
.venv/bin/python scripts/compute_norm_stats.py --config-name pi05_umi_bimanual
.venv/bin/python scripts/train.py pi05_umi_bimanual --exp-name bimanual_run --wandb-enabled
```

Compute stats after editing the dataset/image mapping. This writes
`assets/pi05_umi_bimanual/umi_bimanual/norm_stats.json`. Only dimensions
0,1,2,9,10,11,12,19 are normalized using q01/q99; rotations remain unchanged.
Single-arm normalization assets cannot be reused. The 20-D normalizer explicitly
rejects 10-D quantile arrays. Use a GPU training machine with the existing OpenPI
dependencies and sufficient memory for two camera views.

## 4. Serve the bimanual checkpoint

Choose an existing completed checkpoint step, replacing the example 19999:

```bash
python scripts/serve_policy.py --port 8099 policy:checkpoint \
  --policy.config pi05_umi_bimanual \
  --policy.dir checkpoints/pi05_umi_bimanual/bimanual_run/19999

export OPENPI_DATA_HOME=/share/project/lxy/openpi/.openpi_cache
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
.venv/bin/python scripts/serve_policy.py --port=8099 policy:checkpoint --policy.config=pi05_umi_adjust_tube_20261022 --policy.dir=/share/project/lxy/openpi/checkpoints/pi05_umi_adjust_tube_20261022/adjust_tube/39999

```

No new server script is needed. It loads this data adapter and the checkpoint's
own normalization assets. The adapter uses the following deployment request:

```python
{
    "observation/image_right": right_rgb_uint8,  # [240,240,3]
    "observation/image_left": left_rgb_uint8,    # [240,240,3]
    "current_tcp_pose9_gripper": right_left_reference,  # float32 [20]
    "prompt": "your actual task instruction",
}
```

The reference has the same per-arm order as the action vector. Its poses are
measured absolute TCP poses in each robot's own base, with column-wise rotation6d.
It bypasses model conditioning and anchors the outputs during server postprocessing:

```text
right_target[k] = right_measured_reference @ predicted_right_relative[k]
left_target[k]  = left_measured_reference  @ predicted_left_relative[k]
```

The response is `{"actions": array[32,20]}` (plus existing timing metadata), with
absolute poses in each robot's own base and absolute model-unit gripper targets.
Raw internal model predictions are relative and padded; the server response is
already unnormalized, unpadded, and converted to absolute poses.

## 5. Use the existing BiArm client

Set `policy.server_url` to `ws://SERVER_IP:8099`, set a real prompt, and retain
`action_is_relative: false`. List right hardware first and left second. The
current defaults `observation/image_right` and `observation/image_left` match.
Keep model/client gripper scales and frequency consistent with the dataset.
Do not run the relative-to-absolute transformation a second time in the client.
Start with the existing `--debug-actions --save-policy-data` inspection workflow.
The two-arm checkpoint cannot be served through the single-arm adapter or vice versa.

## Validation performed here

NumPy adapter/geometry/normalization tests are runnable without physical hardware.
On this development host the JAX/Flax/Torch/OpenPI training dependencies are
missing. A dependency-isolated harness ran the existing single-arm and new
bimanual numerical tests using the actual UMI functions; it substituted only
unavailable module interfaces (model enum, transform protocol, stats container,
and CHW rearrange). This is not GPU training, a model forward pass, or a loaded
checkpoint/server test. Run the full test command above in your training environment.
The config integration tests are included for that environment.

Local results: 16 numerical tests passed, including the existing single-arm
regression tests. Isolated config-factory and dataloader dispatch checks passed;
a request built by the current BiArm client was accepted and a synthetic absolute
response decoded correctly by that client. Python syntax and pyflakes checks
passed. These checks do not substitute for the full dependency-backed config
tests, real LeRobot loading, GPU training, or checkpoint inference.

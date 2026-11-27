# RTC deployment for the existing UMI checkpoints

This adds inference-time Real-Time Chunking (RTC) to the two copies in this folder. Existing training code, model parameters, checkpoint format, `Pi0.sample_actions(...)`, hardware drivers, YAML files, and non-RTC clients are unchanged. No retraining is required for this implementation.

**Scope:** standard **JAX Pi0.5 UMI** checkpoints, with one or two arms. This implementation rejects PyTorch and other model classes; it does not silently run ordinary inference when RTC is requested. It has offline tests, but has not been run with your checkpoint/GPU or robots. RTC guidance increases inference compute and needs measurement on the deployment server.

## Added files

Paths below are relative to `RTC-deploy/`.

| File | Purpose |
| --- | --- |
| `openpi-train-RTC/src/openpi/models/rtc_sampler.py` | Separate flow sampler with denoiser-vector-Jacobian-product guidance, following RTC equations 2–5. Leaves the original sampler and learned weights unchanged. |
| `openpi-train-RTC/src/openpi/policies/rtc_umi.py` | Optional RTC request adapter; re-expresses the retained absolute plan in each new observed TCP frame, normalizes using the checkpoint's statistics, runs guidance, and uses the existing output transforms. |
| `openpi-train-RTC/scripts/serve_policy_rtc.py` | Loads an existing checkpoint and serves it using the existing WebSocket transport. |
| `omni-client-lxy-BiArm-RTC/umi/real_world/rtc_policy.py` | Timestamped plan, overlap weights, transport, robot scheduling, and timed custom-gripper dispatch. |
| `omni-client-lxy-BiArm-RTC/policy_client_pi_rtc.py` | Separate episode client: asynchronous inference, continuous short-horizon submission, C/S/R/Q controls, timing logs and optional saved arrays. |
| Both repositories' `tests/test_rtc_*.py` | Offline timing, calibration, geometry, and numerical guidance tests. |

## What RTC changes

The previous client waited for inference and then submitted selected chunk rows. This client keeps a timestamped absolute plan, requests the next prediction in a background thread, and continues submitting the current plan while inference runs.

1. Read the newest observation and its timestamp, just as in your current client.
2. Resample the previous absolute plan onto the new request's grid: `observation_time + k / frequency`.
3. Set the prefix weights to 1 through the estimated inference-plus-delivery delay. Weights decay over the remaining old-plan overlap, and are zero beyond that overlap. A first request uses the measured pose as an initial hold plan.
4. On the server, independently for each arm, compute:

   ```text
   relative_prefix[k] = inverse(T_base_current_tcp) @ T_base_retained_target[k]
   ```

   Encode rotation as the first two rotation-matrix columns, apply the same xyz/gripper quantile normalization as the checkpoint, and zero the guidance mask on padded action dimensions. Gripper targets remain absolute model-space values.
5. Apply guidance inside the flow sampler. The image/text/state adapters and learned network remain the same. The server's existing output transforms return absolute targets in the corresponding arm's robot base.
6. Accept a response only if its first replaceable timestamp is still sufficiently far in the future. Already submitted robot targets are never overwritten. Late responses are discarded, and measured request latency updates the next delay estimate.
7. Submit one waypoint per control tick, just far enough ahead for the configured hardware latency and delivery margin. The existing robot worker still interpolates at its unchanged internal frequency. Keep future custom-gripper commands locally until their compensated due time.

Guidance is a soft constraint, including where the prefix weight is 1; it is not an exact equality guarantee. The client preserves already committed targets independently of the guidance result. RTC reduces chunk-switch discontinuities but cannot fix calibration, frame errors, unstable payload settings, or inaccurate model predictions.

The implementation follows [Real-Time Chunking, arXiv:2506.07339](https://www.pi.website/download/real_time_chunking.pdf). OpenPI integrates from noise at `t=1` to data at `t=0`, so the guidance sign is reversed relative to the paper's increasing-time convention.

## Model/API compatibility

The existing fields are unchanged:

- Single arm: `observation/image` plus `current_tcp_pose9_gripper` of shape `[10]`.
- Two arms: `observation/image_right`, `observation/image_left`, and a single state/reference vector of shape `[20]`, right first.
- RGB images follow your existing camera/ROI path, then the existing client resizing (default 240×240) and server preprocessing.
- Each reference/action block is `[xyz_m(3), R[:,0](3), R[:,1](3), gripper_model_value(1)]`.
- Each response remains `actions: [32,10]` or `[32,20]`, in absolute robot-base coordinates. The learned network retains its existing padded dimension 32.
- Per-gripper `width_to_rad` calibration, including its offset, is reused in both directions. No new common gripper scale is assumed.

RTC adds **optional transport metadata**, not new learned observation inputs:

```text
rtc.version          = 1
rtc.prefix_absolute  = [32, 10*N] absolute targets, same units as regular responses
rtc.weights          = [32] values in [0,1]
```

The new server acknowledges `rtc_version: 1`. An ordinary request without `rtc` uses the original `base.infer` path. The new RTC client refuses a server lacking RTC metadata/acknowledgement. Thus the model's architecture and existing sampling API stay unchanged, but both new deployment entrypoints are needed; an unchanged server cannot provide RTC guidance.

The existing training convention (relative to the first action in each training chunk) is untouched. RTC uses the existing deployment convention (relative to the request observation). This adds no new training format and does not resolve any pre-existing dataset timestamp/reference misalignment. Absolute references still serve geometry conversion rather than introducing measured proprioception into your vision-only adapter.

## Starting the server

Use your working OpenPI server environment on the inference machine. Copy the added server files into the corresponding paths there. Start from `openpi-train-RTC/` so imports resolve to this copy:

```bash
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
python scripts/serve_policy_rtc.py \
  --config pi05_umi_bimanual \
  --checkpoint /absolute/path/to/your/existing/checkpoint \
  --port 8099 \
  --fps 30 \
  --num-steps 10 \
  --max-guidance 5
```

Use the **same training config/asset ID as your checkpoint**, not necessarily the example name above. For an existing single-arm lift-bottle checkpoint, use its original config such as `pi05_umi_lift_bottle`. Statistics load from that checkpoint's `assets/<asset_id>`; do not substitute another task's statistics.

`--fps 30` is only correct if the checkpoint's action sequence represents 30 Hz data. Supply the actual training sampling rate. Your copied bimanual YAML currently says `control_fps: 15`, while the generic training metadata says 30; neither proves a particular checkpoint's dataset rate. Check that dataset/config, then use the same value on both sides. This value sets model action timestamps and does not change the Franka worker's internal frequency.

## Starting the client

Use your existing hardware environment, from `omni-client-lxy-BiArm-RTC/`. Keep the existing YAML format and fill the robot IP/port pairs, camera serials, separate gripper ports, calibration, initial positions, prompt, and `policy.server_url`. Set `policy.action_is_relative: false`: the server already converts its output to absolute targets.

First run prediction with virtual execution (no `--execute`):

```bash
python policy_client_pi_rtc.py \
  --robot-config example/eval_pi05_bimanual_pick_cup_rod.yaml \
  --output data_local/rtc_debug_001 \
  --frequency 30 \
  --steps-per-inference 8 \
  --initial-delay 0.20 \
  --delay-margin 0.05 \
  --delivery-margin 0.05 \
  --save-policy-data
```

Use a fresh output directory each run. Again, replace `30` with the actual action rate. For single-arm use, supply a valid one-arm YAML; arm count is inferred from the hardware lists, exactly as in your existing pipeline.

**Debug mode still starts the existing hardware environment**, which enables holding controllers and resets the custom gripper's reference through its existing startup path. It suppresses policy targets, joint homing, and initial opening commands; it is not a hardware-disconnected test. The pytest tests below are hardware-disconnected.

The client performs two inference warmups before accepting C, without executing their returned actions. This removes the initial JAX compilation delay from the first episode. A first compilation may exceed the existing YAML `policy.request_timeout_s`; increase that timeout if necessary. It is separate from the RTC delay estimate.

After reviewing saved targets/timings, use the same command with **`--execute`** and a new output directory for live execution. With live execution, configured `joints_init` values trigger the existing startup initialization path, and configured gripper `initial_position` values are restored. No new gripper zeroing behavior is added.

Keys (case insensitive, terminal input may require Enter):

- **C:** start an episode from the current pose; no automatic home on C.
- **S**, duration limit, or step limit: stop issuing new targets, let the short already-committed horizon finish, then wait at the last target. S is not an emergency stop.
- **R** while idle: use the existing joint-return helper and restore initial gripper openings without resetting zeros. If YAML has no initial joints, use the measured startup joints.
- **Q:** end the episode and exit.

A pending WebSocket result is discarded before another episode or R can reuse the connection. Waiting for this result can delay the idle prompt up to the configured network timeout; no new policy targets are issued during that wait. Protocol/network/control-loop deadline failures exit with an error rather than silently continuing.

## Parameters and diagnostics

| Setting | Meaning |
| --- | --- |
| `--frequency` | Timestamp spacing of model actions; defaults to YAML `policy.control_fps`. Must match server `--fps` and the actual trained action rate. |
| `--steps-per-inference` | Minimum interval between request starts in control steps. Inference can still take longer. Execution continues while the request is pending. |
| `--initial-delay` | Initial estimate of steady-state RTC request/inference latency, after warmup. Choose it from measured RTC latency, not ordinary inference latency. |
| `--delay-margin` | Added to measured round-trip latency when updating the rolling maximum estimate. |
| `--delivery-margin` | Extra lead beyond the largest configured robot/gripper action latency. The client accounts for hardware latency before accepting a response or scheduling targets. |
| `--max-hold` | Maximum extension of the last accepted plan beyond its final timestamp, default 0.5 s, before ending with an error. |
| `--max-duration`, `--max-timesteps` | Episode duration and number of committed control ticks, including initial holds. |
| Server `--max-guidance` | Upper bound on RTC guidance coefficient (default 5); not a velocity limit. |
| Server `--num-steps` | Flow integration steps (default 10); changes inference compute/quality, not control frequency. |

The new client intentionally has no `--exec-horizon-steps` or `action_exec_latency` override. Existing YAML `exec_horizon_steps` is unused by this separate client. It retains the full predicted chunk in its local buffer and submits only the short committed horizon. It does not enqueue 32 targets at every response or temporally average chunks.

`rtc_debug.jsonl` records request observation times, estimated delay, prefix length (`frozen`), response latency, acceptance/rejection, collision correction magnitude, and every submitted/virtual action with its timestamp. `frozen` counts rows retained from the previous plan; it is not the old stale-action filtering count. With `--save-policy-data`, NPZ files contain RGB images, state/reference, absolute prefix, weights, raw server actions, and request timestamps. Synchronous data saving can itself add latency; disable it when measuring final deployment timing.

If many responses are rejected, first inspect measured RTC request latency and observation age. Increase the initial estimate/margin only enough to cover those delays. If observation age + inference + delivery lead consumes all 32 steps, the client errors rather than retiming old targets. A faster inference server, fewer flow steps after validating quality, or a suitable existing longer-horizon checkpoint is needed; arbitrarily lowering action frequency changes the learned trajectory's time scale.

Existing table/sphere corrections are reused from `policy_client_pi_episodes.py`. In particular, its inter-base sphere transform remains the existing hard-coded translation `[0, 0.635, 0]`. Verify it for your mounting. RTC does not add collision guarantees or velocity limits.

## Offline verification

Client environment, from the client directory:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python -m pytest -q tests/test_rtc_policy.py
```

Server environment, from the server directory with `PYTHONPATH` set as above:

```bash
python -m pytest -q tests/test_rtc_geometry.py tests/test_rtc_sampler.py
```

These tests use small arrays/fake hardware and numerical JAX functions. They do not load your checkpoint or measure GPU latency. A real checkpoint warmup, debug capture inspection, and measured timing remain required before assessing deployment behavior.

### Checks completed during implementation

- 9 client tests passed, including an asynchronous simulated episode, late-response checks, gripper calibration/dispatch, and preservation of committed targets.
- 5 sampler tests passed on CPU with JAX 0.5.3 and Flax 0.10.5, including guidance direction/Jacobian checks, zero-guidance equivalence on a toy velocity model, and binding through the repository's actual `module_jit` without changing toy-model parameters.
- 4 geometry/adapter tests passed for one and two arms. They executed the repository's geometry and normalization code; a local test harness supplied only the transform interface/type definitions to avoid importing unavailable tokenizer/training dependencies. The real checkpoint input/tokenizer pipeline was not exercised.
- All added Python files were syntax-checked. Hash comparison confirmed all 1,146 pre-existing Python/YAML files were unchanged.

The isolated test dependencies were installed under `/tmp`, not into your working Python environments. No robot, motor, or camera was connected during these checks.

## Dedicated RTC client YAML

Use `omni-client-lxy-BiArm-RTC/example/eval_pi05_rtc.yaml` with `--robot-config`.
It copies the existing bimanual hardware/task settings (including 15 Hz) and adds
an `rtc` mapping for `steps_per_inference`, `initial_delay`, `delay_margin`,
`delivery_margin`, `max_hold`, `max_duration`, and `max_timesteps`. Verify the
rate against your checkpoint before running. This template does not change the
robot worker frequency. The obsolete `policy.exec_horizon_steps` is omitted.

Precedence is **explicit CLI argument > YAML > existing built-in default**.
`policy.control_fps` and `policy.image_resolution` supply frequency and image size.
Old YAML without `rtc` continues to work. `--execute`, `--init-joints`, and
`--save-policy-data` remain command-line switches; YAML cannot enable live motion.
Server checkpoint/config, flow steps, and guidance coefficient remain server CLI
settings. The template comments include an example launch command.

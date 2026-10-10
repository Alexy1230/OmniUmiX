# Isolated MVS service and standalone motor test

This change targets `Data-Scaling-Laws-client-lxy`. Its reference is
`OmniUmiX/scripts_infer/camera_node_launcher.py`: the launcher calls
`subprocess.Popen` to start the camera publisher as a new Python interpreter.
Inside that publisher, `MVSCamController` inherits from `BaseController(mp.Process)`.
The device-mapping server runs in a thread; acquisition is in separate processes.

## Why this strategy

The previous timeout says only that the camera did not finish its first frame
publication. It does not prove that GetImageBuffer failed or that the SDK is bad.
The independent camera test succeeding suggests process/library initialization
could matter, but this has not been established on the actual hardware.

The environment now starts a fresh camera-only Python service using
`subprocess.Popen([sys.executable, ...])`. That service owns the same
`MultiMvsCamera`, SDK adapter, image transforms, shared buffers and recorder as the
successful camera-only test. The policy process and preview process retrieve
frames through a private local Unix socket. Observations retain RGB uint8 (or
configured float32), shape, history and original capture/receipt timestamps;
preview/recording retain BGR. There is no JPEG compression. IPC adds copies and
latency compared with directly shared images; measure FPS/observation latency on
the deployment PC, especially at large resolutions or with multiple cameras.

No ROS installation or global multiprocessing start-method change is required.
Do not add `spawn`, camera startup inside BimanualUmiEnv.__init__, the temporary
manager pickling change, or the previous lazy-import workarounds to this baseline.
`uvc_camera.py`, `mvs_camera.py`, `mvs_capture.py`, and both policy clients are
unchanged by this patch. It changes how the environment obtains its camera.

## Files changed (copy all of them)

- `umi/real_world/mvs_camera_service.py` (new): fresh-interpreter launcher, bounded
  startup/RPC waits, local observation/preview/recording API, error forwarding,
  stale-frame detection, owned-process cleanup.
- `umi/real_world/bimanual_umi_env.py`: select IsolatedMultiMvsCamera for MVS;
  remove the unconditional `ready_flag = True` so readiness reflects the workers.
- `scripts/test_mvs_camera.py`: add `--isolated`; the default direct test remains.
- `scripts/test_gripper_motor.py` (new): read-only serial feedback test, optional
  explicitly requested movement, fresh-feedback and range checks, serial cleanup.
- `umi/real_world/gripper_controller_1_14.py`: add `start(reset_zero=True)` (old
  callers keep their existing zero-reset behavior); optional False skips resetting
  the encoder zero; seed the first commanded target ramp at measured position.
- `tests/test_mvs_service.py`, `tests/fake_mvs_service.py`,
  `tests/test_gripper_motor_check.py`: hardware-free tests.
- This document.

Use the project Python environment which already runs the direct camera test.
The service uses that exact same `sys.executable` and the project's absolute path.
The motor test needs the existing `motor.motor` package used by
`gripper_controller_1_14.py`; it does not import a Franka gripper module.

## Camera test order

Run from the project root, with each previous camera test/viewer fully closed:

```bash
python scripts/test_mvs_camera.py --config example/eval_pi05_adjust_bottle_mvs.yaml --seconds 10 --show
python scripts/test_mvs_camera.py --config example/eval_pi05_adjust_bottle_mvs.yaml --isolated --seconds 10 --show
```

To exercise recording too:

```bash
python scripts/test_mvs_camera.py --config example/eval_pi05_adjust_bottle_mvs.yaml --isolated --seconds 10 --output data_local/isolated_camera_check
```

Then use the existing zero-action client (the actual file in this repository is
`zero_action_client_pi.py`):

```bash
python zero_action_client_pi.py --robot-config example/eval_pi05_adjust_bottle_mvs.yaml --server-url ws://SERVER_IP:8099 --output data_local/pi05_zero_action
```

Expect `MVS service ready (PID ...)` before the robot workers start. The SDK runs
under that service. Camera shutdown normally flushes/stops the recorder, closes
the SDK and exits. If startup/shutdown is stuck, the proxy terminates its isolated
process group. It never kills unrelated Python processes. Service log output is
included in timeout/crash exceptions; temporary files are removed on cleanup.
The local socket lives in a mode-0700 temporary directory. Its pickle protocol
is local to trusted processes of the current user; do not expose it over a network.

## Motor connection test (no movement)

```bash
python scripts/test_gripper_motor.py --config example/eval_pi05_adjust_bottle_mvs.yaml --seconds 5
```

Overrides when the serial device name changes:

```bash
python scripts/test_gripper_motor.py --config example/eval_pi05_adjust_bottle_mvs.yaml --port /dev/ttyUSB1 --motor-id 2 --seconds 5
```

The default mode sends feedback-read requests only. It never calls start(),
enable(), set_zero_ram(), move_gripper(), or disable(). It closes the serial port
on exit. It prints motor angle, driver width, velocity and torque. A failed or
non-finite response fails the test instead of displaying a fabricated zero.
Close the policy client before this test so only one program owns the serial port.

## Explicit movement test

`--move-mm` is the ABSOLUTE width according to the existing driver's encoder zero
and 30 mm/rad conversion. It is not a relative step, and an uncalibrated zero is
not a physical jaw-width measurement. First read the current driver width.
For example, if that read is 40 mm, request 45 mm:

```bash
python scripts/test_gripper_motor.py --config example/eval_pi05_adjust_bottle_mvs.yaml --move-mm 45 --seconds 10 --torque-limit 0.3 --speed-mm-s 5
```

Movement uses the existing GripperController control loop and move_gripper API.
It enables the motor without changing its zero and disables it on exit. The
maximum requested travel defaults to 10 mm; target range is also checked against
the driver's limit. A target must be reached within 1 mm on three successive
feedback samples, otherwise the test fails after the time limit. No automatic
return-to-start movement is issued. Adjust the target and limits deliberately.

In the current driver, the move_gripper force_limit argument is not used for the
actual clipping of commanded torque. This test sets TORQUE_LIMIT_SINGLE directly
to the requested limit before enabling the controller. This is a command limit,
not a calibrated gripping-force measurement.

## Validation

```bash
python tests/test_mvs_service.py
python tests/test_gripper_motor_check.py
```

These tests use a fake camera service and fake serial motor. They cover actual
subprocess/socket communication, history/array preservation, concurrent readers,
recording-command forwarding, startup failure, timeout cleanup, crash handling,
read-only behavior, invalid feedback/targets, optional zero preservation, and
initial hold-position ramp behavior. They do not validate the MVS SDK, USB link,
physical motor calibration, or achievable frame rate on the deployment PC.

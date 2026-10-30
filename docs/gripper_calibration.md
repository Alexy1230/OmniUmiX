# Per-arm gripper calibration

Add `width_to_rad` inside each entry of the existing `grippers` list. List order remains right, then left; one entry works for single-arm use. Copy the actual calibration produced for that gripper, rather than the illustrative numbers below:

```yaml
grippers:
  - gripper_type: custom
    port: /dev/serial/by-id/RIGHT_ADAPTER
    motor_id: 1
    mode: single
    initial_position: 0.045
    width_to_rad:
      min_abs_width_m: 0.010
      max_abs_width_m: 0.080
      min_rad: 0.0
      max_rad: 3.5
    # Keep your other existing settings here.
  - gripper_type: custom
    port: /dev/serial/by-id/LEFT_ADAPTER
    motor_id: 1
    mode: single
    initial_position: 0.060
    width_to_rad:
      min_abs_width_m: 0.020
      max_abs_width_m: 0.100
      min_rad: 0.0
      max_rad: 3.5
    # Keep your other existing settings here.
```

For width w (metres) and model/motor angle r (radians):

```
s = (max_rad - min_rad) / (max_abs_width_m - min_abs_width_m)
b = min_rad - s * min_abs_width_m
r = s * w + b
w = (r - b) / s
```

Both `policy_client_pi.py` and `policy_client_pi_episodes.py` derive these independently per arm. Calibrated entries override policy-level hardware_to_model_gripper_scale and model_to_hardware_gripper_scale. Those legacy settings remain fallbacks for entries without calibration. Offsets are included in saved policy metadata/debug options. `initial_position` remains metres and must lie within the supplied width interval.

Predicted widths are clipped to the calibrated interval; measured readings are not clipped so out-of-range feedback stays visible. The local custom motor controller uses the same calibration in both directions, including width history and startup opening. The existing 0.20 m driver ceiling remains in force. This avoids applying a new model mapping on top of the controller's old fixed 30 mm/rad mapping. WSG controllers already report widths in metres and need no driver conversion changes.

The min_rad/max_rad endpoints must correspond to the actual motor-angle reference as well as the dataset's model-angle convention. This configuration does not physically home the motor or discover its zero. Existing controller startup still sets the current motor position to zero. With min_rad: 0, that startup position must physically correspond to min_abs_width_m for the calibration to hold. It is not necessarily zero millimetres. Do not assume an arbitrary startup opening is the calibrated zero. Initial opening commands happen after this existing zero-setting step.

For the motor test, use the same YAML; gripper index defaults to 0 (right), and 1 selects left:

```bash
python scripts/test_gripper_motor.py --config example/eval_pi05_bimanual_mvs.yaml --gripper-index 1 --seconds 5
```

The test's default mode only reads feedback. Existing movement options remain explicit. It now uses calibration and accepts the current port/motor_id keys as well as legacy port1/motor1_id keys.

Files changed: shared conversion helpers, the two current clients, local gripper controller, environment constructor wiring, motor test, commented YAML examples, and offline tests. Historical `policy_client_pi_dangerous.py` and `policy_client_pi_episodes_test.py` are not updated deployment entrypoints. No external motor package was modified.

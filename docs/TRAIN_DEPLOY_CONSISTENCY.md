# FOV167 collection/deployment contract

The client must send a 240x240 RGB image. OpenPI resizes that square image to
224x224 inside the model pipeline during both training and inference. The
240-to-224 resize is therefore expected and must not be replaced by a crop or
pad in only one side of the pipeline.

## Task-specific deployment camera geometry

Deployment must follow the crop mode used to build each training dataset:

1. acquire the native 1440x1080 frame;
2. apply the configured 180-degree mounting rotation;
3. for `stretch`, keep the full 1440x1080 frame; for `center_square`, crop
   columns `[180:1260]` to 1080x1080 at center `(720, 540)`;
4. resize once to 240x240 with FFmpeg/libswscale Lanczos;
5. convert BGR to RGB before sending it to the policy server.

`adjust_tube_fov167` uses `stretch` because its acquisition task has
`crop: false` and its active processing config explicitly selects `stretch`.
Tasks collected with `crop: true` use `center_square` when processing uses
`crop_mode: auto`.

The profiles declare `source_width`, `source_height`, ROI, output resolution,
and interpolation explicitly. The collection repository and existing datasets
are not modified by this deployment contract.

## Other verified conventions

- Collection stores upright RGB frames; the deployment driver acquires BGR,
  rotates by 180 degrees, applies the task's crop mode, then converts to RGB.
- Single-arm state/action blocks are pose10: xyz, two rotation-matrix columns,
  and gripper radians. Actions are absolute for these profiles.
- The gripper `width_to_rad` endpoints come from the calibration archived with
  the datasets. `--gripper-close-offset-mm` is an optional execution tuning on
  top of that calibration and defaults to zero.
- Training data is 30 Hz. A lower execution frequency is an intentional time
  scaling decision, not an image preprocessing change.
- A training config may override dataset task text with `default_prompt`.
  Confirm the exact server checkpoint config before changing a deployment
  prompt solely to match `meta/tasks.jsonl`.

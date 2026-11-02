# Omni robot policy clients

This repository contains robot policy clients, camera services, calibration tools,
and shared UMI runtime utilities. Run commands from
this directory so local packages and relative configuration paths resolve.

## Start here

- [Inference guide](docs/inference.md)
- [Proprioception client guide](docs/PROPRIO_CLIENT.md)
- [Isolated camera and motor checks](docs/isolated_mvs_and_motor_test.md)
- [Gripper calibration](docs/gripper_calibration.md)

`conda_environment.yaml` records the inherited environment. Hardware SDKs,
robot connections, camera selection, and policy-server settings depend on the
chosen client; follow the guides and inspect the matching configuration under
`example/` before running hardware commands.

## Client variants

All current variants are retained: `policy_client_pi.py`,
`policy_client_pi_wolatency.py`, `policy_client_pi_episodes.py`,
`policy_client_pi_proprio_episodes.py`, `policy_client_gripper_tuned.py`,
`zero_action_client_pi.py`, `umi_policy_client.py`, and
`example/continuous_client.py`. Their names alone do not establish obsolescence.
`teleop_interface.py` provides a separate teleoperation entry point.

## Repository layout

- Root `policy_client*.py`: Pi model clients (vision and proprio are separate entry points).
- `umi/real_world/`: camera and robot runtime; `umi/shared_memory/`: IPC.
- `umi/compat/`: small upstream utility subset required for recording and pose math; no DP/RDP policy models.
- `scripts/`: retained camera, motor and calibration diagnostics.
- `scripts_real/`: retained hardware control and demonstrations.
- `example/`: robot/task YAMLs and calibration files.
- `tests/`: automated checks and separate hardware diagnostics.

## Package cleanup

The complete diffusion_policy and reactive_diffusion_policy packages, including their scripts copies,
were removed. Required helpers were migrated into umi/compat with imports updated.
Old local-policy evaluators, training launcher and selected SLAM/video-processing scripts were removed.
See docs/cleanup/pi_packages_removed.json for every removed/migrated/updated file.
Earlier cleanup reports are historical. Configurations, logs and LICENSE were retained.
Recovery archive: ../omni-client-cleanup-backup-20261102.tar.gz (not part of this repo).

Run targeted tests only: some diagnostic scripts connect to hardware.

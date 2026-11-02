# Follow-up validation — 2026-11-02

The working tree already contained the cleanup before this follow-up inspection. Its manifest lists 796 deletions; these were not repeated or attributed to new work. Existing modifications were preserved.

82 tests passed across test_proprio_client.py, test_bimanual_client.py, test_policy_request_keys.py, test_gripper_calibration.py, test_policy_action_timing.py and test_wolatency_gripper_offset.py. Tests ran without connecting to hardware.

Reviewed existing fixes: bimanual_policy rejects overlapping image/state/prompt keys; bimanual_umi_env filters targets using the maximum applicable delivery latency before compensating timestamps. These changes were already in the working tree.

README corrected: train.py/eval_real.py still exist, so not all legacy launchers have been removed. Replaced a broken audit-document link with closure.json.

Retained all client variants, hardware utilities, configurations, logs and LICENSE. Dependency closure is static and cannot prove all dynamic imports or external use. No claim of zero bugs, hardware validation, complete legacy removal, GPU execution, or dual-arm proprio model compatibility is made.

Full deletion inventory: manifest.md and deletions.json. A temporary backup from the earlier cleanup exists at /tmp/omni-cleanup/v2/backup.tar.gz; do not rely on /tmp for long-term retention. No commit or push was performed.

import os
import pathlib
import sys

import numpy as np


os.environ.setdefault('PYNPUT_BACKEND', 'dummy')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from policy_client_pi_episodes_test import (  # noqa: E402
    TimeAlignedActionEnsembler,
    compute_latency_trimmed_steps,
    reanchor_action_chunk,
)


def test_latency_trim_and_reanchor_retain_remaining_motion():
    dt = 1.0 / 30.0
    trim = compute_latency_trimmed_steps(
        observation_timestamp=100.0,
        planning_time=100.15,
        execution_margin=0.15,
        dt=dt,
        horizon=32,
    )
    assert trim == 9

    actions = np.zeros((32, 7), dtype=np.float64)
    actions[:, 0] = np.arange(32) * 0.002
    anchor = np.array([0.5, 0.2, 0.4, 0.0, 0.0, 0.0])
    execution = reanchor_action_chunk(actions[trim:], anchor)

    np.testing.assert_allclose(execution[0, :6], anchor, atol=1e-12)
    assert len(execution) == 23
    np.testing.assert_allclose(
        execution[-1, 0] - execution[0, 0],
        (31 - trim) * 0.002,
        atol=1e-12,
    )


def test_observation_timing_aligns_overlapping_policy_frames():
    ensembler = TimeAlignedActionEnsembler(
        dt=1.0,
        time_origin=0.0,
        max_chunks=3,
        consensus_translation_m=1.0,
    )
    older = np.zeros((8, 7), dtype=np.float64)
    older[:, 0] = np.arange(8)
    newer = np.zeros((8, 7), dtype=np.float64)
    newer[:, 0] = 2.0 + np.arange(8)

    ensembler.add_chunk(older, observation_timestamp=0.0)
    ensembler.add_chunk(newer, observation_timestamp=2.0)
    fused, timestamps, contributors = ensembler.get_fused_trajectory(
        after_time=3.5,
        max_steps=3,
    )

    np.testing.assert_allclose(timestamps, [4.0, 5.0, 6.0])
    np.testing.assert_allclose(fused[:, 0], [3.0, 4.0, 5.0])
    np.testing.assert_array_equal(contributors, [2, 2, 2])

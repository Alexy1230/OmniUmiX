"""Checks for physical units, arm ordering, and exclusion of padded timesteps."""

import numpy as np

from scripts.eval_umi_bimanual import chunk_errors
from scripts.eval_umi_bimanual import plot_chunk
from scripts.eval_umi_bimanual import summarize_errors


def test_errors_separate_arms_and_use_physical_units():
    target = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 2, (3, 1)).astype(np.float32)
    prediction = target.copy()
    prediction[:, 0] += 0.01
    prediction[:, 11] += 0.02
    prediction[:, 9] += 0.3
    prediction[:, 13:19] = [0, 1, 0, -1, 0, 0]  # left rotates 90 degrees about z
    errors = chunk_errors(prediction, target)
    np.testing.assert_allclose(errors["right_position_mm"], 10, atol=1e-5)
    np.testing.assert_allclose(errors["left_position_mm"], 20, atol=1e-5)
    np.testing.assert_allclose(errors["right_rotation_deg"], 0, atol=1e-5)
    np.testing.assert_allclose(errors["left_rotation_deg"], 90, atol=1e-5)
    np.testing.assert_allclose(errors["right_gripper_abs"], 0.3)
    np.testing.assert_allclose(errors["left_gripper_abs"], 0)


def test_summary_weights_valid_timesteps_and_excludes_padding():
    chunks = [{"error": np.array([1.0, 2.0, 1e6])}, {"error": np.array([3.0, 1e6, 1e6])}]
    masks = [np.array([True, True, False]), np.array([True, False, False])]
    result = summarize_errors(chunks, masks)["error"]
    np.testing.assert_allclose(result["mean"], 2)
    np.testing.assert_allclose(result["rmse"], np.sqrt(14 / 3))
    np.testing.assert_allclose(result["p95"], 2.9)


def test_plot_supports_single_valid_tail_frame(tmp_path):
    target = np.tile([0, 0, 0, 1, 0, 0, 0, 1, 0, 0] * 2, (32, 1)).astype(np.float32)
    valid = np.arange(32) == 0
    path = tmp_path / "tail.png"
    plot_chunk(path, target, target, valid, chunk_errors(target, target), 30, "Tail frame")
    assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

"""Offline tests for the TCP-local Z action offset."""

import ast
from pathlib import Path
import types

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parents[1]


def load_z_tuner():
    tree = ast.parse((ROOT / 'policy_client_gripper_tuned.py').read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == 'apply_tcp_z_forward_offset'
    )
    namespace = {
        'np': np,
        'st': types.SimpleNamespace(Rotation=Rotation),
    }
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'z_tuner', 'exec'), namespace)
    return namespace['apply_tcp_z_forward_offset']


def test_z_offset_uses_each_tcp_local_frame_without_mutating_input():
    tune = load_z_tuner()
    actions = np.zeros((1, 14), dtype=float)
    actions[0, 3:6] = Rotation.from_euler('y', 90, degrees=True).as_rotvec()
    actions[0, 6] = 0.08
    actions[0, 7:10] = [1.0, 2.0, 3.0]
    actions[0, 10:13] = Rotation.from_euler('x', 90, degrees=True).as_rotvec()
    actions[0, 13] = 0.06
    original = actions.copy()

    tuned = tune(actions, 0.01)

    np.testing.assert_allclose(tuned[0, :3], [0.01, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(tuned[0, 7:10], [1.0, 1.99, 3.0], atol=1e-12)
    np.testing.assert_allclose(tuned[0, 3:7], original[0, 3:7])
    np.testing.assert_allclose(tuned[0, 10:14], original[0, 10:14])
    np.testing.assert_array_equal(actions, original)


def test_zero_z_offset_preserves_actions():
    tune = load_z_tuner()
    actions = np.arange(21, dtype=float).reshape(3, 7)
    actions[:, 3:6] = 0.0
    np.testing.assert_array_equal(tune(actions, 0.0), actions)


@pytest.mark.parametrize('offset', [-0.001, np.nan, np.inf])
def test_invalid_z_offset_rejected(offset):
    tune = load_z_tuner()
    with pytest.raises(ValueError, match='finite and non-negative'):
        tune(np.zeros((1, 7)), offset)


def test_invalid_action_shape_rejected():
    tune = load_z_tuner()
    with pytest.raises(ValueError, match=r'7 \* num_robots'):
        tune(np.zeros((7,)), 0.01)
    with pytest.raises(ValueError, match=r'7 \* num_robots'):
        tune(np.zeros((2, 8)), 0.01)

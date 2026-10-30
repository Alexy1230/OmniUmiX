"""Offline tests for the sequential client's gripper output offset."""

import ast
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_tuner():
    tree = ast.parse((ROOT / 'policy_client_pi_wolatency.py').read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == 'apply_gripper_close_offset'
    )
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'gripper_tuner', 'exec'), namespace)
    return namespace['apply_gripper_close_offset']


def load_horizon_selector():
    tree = ast.parse((ROOT / 'policy_client_pi_wolatency.py').read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef) and item.name == 'select_action_horizon'
    )
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'horizon_selector', 'exec'), namespace)
    return namespace['select_action_horizon']


def test_offset_closes_each_gripper_and_does_not_mutate_input():
    tune = load_tuner()
    actions = np.zeros((2, 14), dtype=float)
    actions[:, 6] = [0.080, 0.046]
    actions[:, 13] = [0.070, 0.050]
    original = actions.copy()

    tuned = tune(actions, 0.005, [(0.043, 0.150), (0.040, 0.155)])

    np.testing.assert_allclose(tuned[:, 6], [0.075, 0.043])
    np.testing.assert_allclose(tuned[:, 13], [0.065, 0.045])
    np.testing.assert_array_equal(actions, original)


def test_zero_offset_preserves_actions():
    tune = load_tuner()
    actions = np.arange(21, dtype=float).reshape(3, 7)
    np.testing.assert_array_equal(tune(actions, 0.0, [(0.04, 0.15)]), actions)


@pytest.mark.parametrize('offset', [-0.001, np.nan, np.inf])
def test_invalid_offset_rejected(offset):
    tune = load_tuner()
    with pytest.raises(ValueError, match='finite and non-negative'):
        tune(np.zeros((1, 7)), offset, [(0.04, 0.15)])


def test_invalid_shape_or_limit_count_rejected():
    tune = load_tuner()
    with pytest.raises(ValueError, match=r'7 \* num_robots'):
        tune(np.zeros((2, 8)), 0.003, [(0.04, 0.15)])
    with pytest.raises(ValueError, match='one gripper width limit'):
        tune(np.zeros((2, 14)), 0.003, [(0.04, 0.15)])


def test_execution_horizon_selects_prefix_without_mutating_input():
    select = load_horizon_selector()
    actions = np.arange(32 * 7, dtype=float).reshape(32, 7)

    selected = select(actions, 12)

    assert selected.shape == (12, 7)
    np.testing.assert_array_equal(selected, actions[:12])
    selected[0, 0] = -1
    assert actions[0, 0] == 0


def test_execution_horizon_larger_than_prediction_uses_all_rows():
    select = load_horizon_selector()
    actions = np.zeros((8, 7))
    assert select(actions, 16).shape == (8, 7)


@pytest.mark.parametrize('horizon', [0, -1, 1.5, True])
def test_invalid_execution_horizon_rejected(horizon):
    select = load_horizon_selector()
    with pytest.raises(ValueError, match='positive integer'):
        select(np.zeros((32, 7)), horizon)

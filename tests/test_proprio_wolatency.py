"""Offline checks for the sequential proprio client wiring."""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_history_setter(paired_history):
    tree = ast.parse((ROOT / 'policy_client_pi_wolatency.py').read_text())
    node = next(
        item for item in tree.body
        if isinstance(item, ast.FunctionDef)
        and item.name == 'set_proprio_request_history'
    )
    namespace = {'np': np, 'paired_history': paired_history}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), 'history_setter', 'exec'),
        namespace,
    )
    return namespace['set_proprio_request_history']


def test_history_setter_forwards_aligned_pair_and_timestamps():
    previous = [np.zeros((4, 4, 3), dtype=np.uint8)]
    history = np.arange(20, dtype=np.float32).reshape(2, 10)
    paired_calls = []

    def paired_history(obs, resolution, scales, options, count):
        paired_calls.append((obs, resolution, scales, options, count))
        return previous, history

    received = []
    client = SimpleNamespace(set_history=lambda *args: received.append(args))
    obs = {'timestamp': np.array([10.0, 10.1])}
    setter = load_history_setter(paired_history)

    setter(client, obs, 240, [2.0], {'state_key': 'state'}, 1)

    assert paired_calls == [(obs, 240, [2.0], {'state_key': 'state'}, 1)]
    assert received[0][0] is previous
    assert received[0][1] is history
    assert received[0][2:] == (10.0, 10.1)


def test_history_setter_rejects_non_pair_timestamps():
    setter = load_history_setter(lambda *args, **kwargs: ([], np.zeros((2, 10))))
    client = SimpleNamespace(set_history=lambda *args: None)
    with pytest.raises(ValueError, match='exactly two observation timestamps'):
        setter(client, {'timestamp': np.array([1.0])}, 240, [1.0], {}, 1)


def test_proprio_entrypoint_enables_shared_mode():
    tree = ast.parse((ROOT / 'policy_client_pi_proprio_wolatency.py').read_text())
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute)
            and target.attr == 'PROPRIOCEPTION'
            for target in node.targets
        )
    ]
    assert len(assignments) == 1
    assert isinstance(assignments[0].value, ast.Constant)
    assert assignments[0].value.value is True

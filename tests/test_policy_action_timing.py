"""Exercise actual environment scheduling without importing hardware drivers."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_compensated_deadlines_drop_expired_chunks_for_all_arms():
    source = Path(__file__).resolve().parents[1] / 'umi/real_world/bimanual_umi_env.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BimanualUmiEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'exec_actions')
    namespace = {'np': np, 'time': SimpleNamespace(time=lambda: 100.)}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), 'exec'), namespace)
    calls = []
    controller = SimpleNamespace(schedule_waypoint=lambda **kw: calls.append(kw))
    env = SimpleNamespace(is_ready=True, robots=[controller]*2, grippers=[controller]*2,
        robots_config=[{'robot_action_latency': .1}]*2,
        grippers_config=[{'gripper_action_latency': .3}]*2, action_accumulator=None)
    namespace['exec_actions'](env, np.zeros((3,14)), np.array([99.,100.2,100.5]), True)
    assert len(calls) == 4
    np.testing.assert_allclose([c['target_time'] for c in calls], [100.4,100.2]*2)
    assert all(c['target_time'] > 100 for c in calls)

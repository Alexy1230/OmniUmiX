"""Exercise actual environment scheduling without importing hardware drivers."""
import ast
from collections import deque
import math
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def test_compensated_deadlines_drop_expired_chunks_for_all_arms():
    source = Path(__file__).resolve().parents[1] / 'umi/real_world/bimanual_umi_env.py'
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BimanualUmiEnv')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'exec_actions')
    namespace = {'np': np, 'math': math, 'time': SimpleNamespace(time=lambda: 100.)}
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

def test_action_selection_keeps_earliest_uncommitted_rows_and_respects_deadline():
    from umi.real_world.client_deployment import select_live_waypoints

    indices, timestamps = select_live_waypoints(32, 100.0, .05, 100.1, .15, 100.31, 2, 100.4)
    np.testing.assert_array_equal(indices, [7, 8])
    np.testing.assert_allclose(timestamps, [100.35, 100.4])
    indices, _ = select_live_waypoints(4, 100.0, .05, 100.1, .15, 0., 4, 101.0)
    assert not len(indices)


def test_gripper_timed_waypoints_preserve_prefix_and_drop_missed_targets():
    source = Path(__file__).resolve().parents[1] / 'umi/real_world/gripper_controller_1_14.py'
    cls = next(node for node in ast.parse(source.read_text()).body
               if isinstance(node, ast.ClassDef) and node.name == 'GripperController')
    namespace = {'math': math}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), 'exec'), namespace)
    controller_cls = namespace['GripperController']
    grip = controller_cls.__new__(controller_cls)
    grip.mode = 'single'
    grip._width_to_angle = lambda width: width * 10
    grip._waypoint_lock = threading.Lock()
    grip._waypoints = deque()
    grip.command_queue = queue.Queue()
    grip.schedule_waypoints([.1, .2, .3], [1., 2., 3.], force_limit_nm=1.5)
    grip.schedule_waypoints([.4, .5], [2., 4.], force_limit_nm=1.0)
    assert grip._pop_due_waypoint(.9) is None
    assert grip._pop_due_waypoint(1.) == ('move', 1., 1.5)
    assert grip._pop_due_waypoint(3.) == ('move', 4., 1.)
    assert grip.has_pending_waypoints()
    grip._queue_manual_command(('stop',))
    assert not grip.has_pending_waypoints()
    assert grip.command_queue.get_nowait() == ('stop',)
    grip.schedule_waypoints([.1, .2], [2., 3.], force_limit_nm=1.5)
    assert grip._pop_due_waypoint(4.) == ('move', 2., 1.5)
    assert not grip.has_pending_waypoints()


def test_warmup_requests_fresh_observations_without_submitting_actions():
    from umi.real_world.client_deployment import warmup_requests

    observations = iter([object(), object()])
    calls = []
    keys = iter([set(), {'c'}, set()])
    env = SimpleNamespace(grippers=[], grippers_config=[], get_obs=lambda: next(observations))
    assert warmup_requests(env, calls.append, lambda: next(keys))
    assert len(calls) == 2
    assert calls[0] is not calls[1]

def test_sequential_playback_reaches_each_gripper_target_before_returning():
    src = Path(__file__).resolve().parents[1] / 'policy_client_pi_wolatency.py'
    node = next(n for n in ast.parse(src.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'execute_action_chunk')

    class Clock:
        now = 100.0

        def time(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

    clock = Clock()
    calls = []
    drains = []

    def submit(*, actions, timestamps, compensate_latency):
        assert not compensate_latency
        assert actions.shape == (1, 7)
        calls.append((clock.now, actions[0, 6]))
        return timestamps

    namespace = {'time': clock, 'np': np, 'episode_stop_reason': lambda *a: None,
                 'wait_pending_actions': lambda env, end: drains.append(end)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(src), 'exec'), namespace)
    env = SimpleNamespace(exec_actions=submit)
    actions = np.zeros((3, 7))
    actions[:, 6] = [.06, .05, .04]
    step, reason, timestamps = namespace['execute_action_chunk'](
        env, actions, .05, None, clock.now, 10., 0, 3)
    assert step == 3 and reason is None
    np.testing.assert_allclose(calls, [[100., .06], [100.05, .05], [100.10, .04]])
    np.testing.assert_allclose(timestamps, [100.05, 100.10, 100.15])
    assert clock.now == drains[0] == timestamps[-1]


def test_proprio_warmup_passes_aligned_history_and_resets_start_pose(monkeypatch):
    from umi.real_world import bimanual_policy, client_deployment, proprio_policy_client

    frames = iter([10., 11.])
    def get_obs():
        current = next(frames)
        return {'timestamp': [current - .1, current]}

    env = SimpleNamespace(grippers=[], grippers_config=[], get_obs=get_obs)
    monkeypatch.setattr(bimanual_policy, 'build_observation',
                        lambda obs, *a: (obs, 'state', None, None))
    monkeypatch.setattr(proprio_policy_client, 'paired_history',
                        lambda obs, *a, **kw: ('images_history', 'pose_history'))
    calls = []
    client = SimpleNamespace(
        set_history=lambda *args: calls.append(args),
        call_server=lambda images, state: calls.append(images['timestamp'][-1]),
        reset_episode=lambda: calls.append('reset'),
    )
    assert client_deployment.warmup_pi(env, client, 240, [1.], {}, set)
    assert calls == [('images_history', 'pose_history', 9.9, 10.), 10.,
                     ('images_history', 'pose_history', 10.9, 11.), 11., 'reset']

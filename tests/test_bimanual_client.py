"""Offline tests: real PC conversion/environment methods with fake hardware I/O.

AST loading of the environment avoids importing the unavailable Linux RTDE/MVS/
motor packages on a development machine. No robot or serial connection is opened.
"""
import ast
import importlib.util
from pathlib import Path
import sys
import types
import time
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umi.real_world.bimanual_policy import (  # noqa: E402
    PolicyClient, pack_state, unpack_actions, policy_options, decode_actions,
    rotation6d_to_matrix, validate_hardware_config,
    average_actions, build_observation,
)


def load_collision(entry, name):
    tree = ast.parse((ROOT/entry).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    from umi.common.pose_util import pose_to_mat, mat_to_pose
    ns = dict(np=np, st=types.SimpleNamespace(Rotation=Rotation),
              pose_to_mat=pose_to_mat, mat_to_pose=mat_to_pose)
    exec(compile(ast.Module(body=[node], type_ignores=[]), entry, 'exec'), ns)
    return ns[name]


def load_sphere_collision(entry):
    return load_collision(entry, 'solve_sphere_collision')


def contract():
    return policy_options(dict(image_keys=['right_image', 'left_image'],
                               state_key='reference',
                               action_is_relative=True), 2)


def pose9(xyz, rotation):
    return np.r_[xyz, rotation.as_matrix()[:, :2].T.reshape(6)]


@pytest.mark.parametrize('count', [1, 2])
def test_roundtrip_list_order(count):
    states = np.arange(10 * count).reshape(count, 10)
    packed = pack_state(states)
    np.testing.assert_array_equal(unpack_actions(packed[None], count)[0], states)


def test_relative_motion_uses_each_request_pose_and_rotation():
    poses = np.array([[1, 2, 3, 0, 0, np.pi/2], [-2, 4, 1, 0, 0, -np.pi/2]])
    relative = np.array([np.r_[pose9([0.1, 0, 0], Rotation.from_euler('x', 0.2)), 2],
                         np.r_[pose9([0.2, 0, 0], Rotation.from_euler('y', 0.3)), 1]])
    options = contract()
    raw = pack_state(relative)[None]
    result = decode_actions(raw, poses, [0.03, 0.04], True, options).reshape(1, 2, 7)[0]
    np.testing.assert_allclose(result[:, :3], [[1, 2.1, 3], [-2, 3.8, 1]], atol=1e-7)
    np.testing.assert_allclose(result[:, 6], [0.06, 0.04])
    for i in range(2):
        expected = Rotation.from_rotvec(poses[i, 3:]).as_matrix() @ rotation6d_to_matrix(relative[i, 3:9])
        np.testing.assert_allclose(Rotation.from_rotvec(result[i, 3:6]).as_matrix(), expected, atol=1e-7)
    absolute = decode_actions(raw, poses, [0.03, 0.04], False, options).reshape(1, 2, 7)[0]
    np.testing.assert_allclose(absolute[:, :3], relative[:, :3])


def test_single_arm_wire_format_and_decode_unchanged():
    options = policy_options({}, 1)
    base = np.array([[1, 0, 0, 0, 0, 0]])
    raw = np.r_[pose9([0.1, 0, 0], Rotation.identity()), 2][None]
    result = decode_actions(raw, base, [0.03], True, options)
    np.testing.assert_allclose(result, [[1.1, 0, 0, 0, 0, 0, 0.06]])
    assert options['image_keys'] == ['observation/image_right']


def test_request_contains_distinct_images_and_exact_state():
    client = PolicyClient.__new__(PolicyClient)
    client.options, client.num_robots, client.prompt = contract(), 2, 'fold'
    images = [np.zeros((4, 4, 3), np.uint8), np.full((4, 4, 3), 99, np.uint8)]
    payload = client.build_payload(images, np.arange(20))
    assert payload['left_image'][0, 0, 0] == 99  # left image maps to hardware 1
    assert payload['right_image'][0, 0, 0] == 0
    assert payload['reference'].dtype == np.float32
    assert payload['prompt'] == 'fold'


def test_reject_single_arm_output_for_two_arms_and_degenerate_rotation():
    with pytest.raises(ValueError, match='20'):
        unpack_actions(np.zeros((32, 10)), 2)
    with pytest.raises(ValueError, match='rotation'):
        rotation6d_to_matrix(np.zeros((1, 6)))


def configs():
    robots = [dict(robot_type='franka', robot_ip=f'nuc{i}', robot_port=4242,
                   robot_obs_latency=0, robot_action_latency=0.1 + 0.1*i,
                   tx_flange_tcp=np.eye(4).tolist()) for i in range(2)]
    grippers = [dict(gripper_type='custom', port1=f'/dev/hand{i}', motor1_id=1,
                     mode='single', gripper_obs_latency=0.01, gripper_action_latency=0.1)
                for i in range(2)]
    return robots, grippers


def test_duplicate_endpoints_and_ports_rejected():
    robots, grippers = configs()
    validate_hardware_config(robots, grippers)
    robots[1]['robot_ip'] = robots[0]['robot_ip']
    with pytest.raises(ValueError, match='duplicate robot endpoint'):
        validate_hardware_config(robots, grippers)
    robots[1]['robot_port'] = 4243
    validate_hardware_config(robots, grippers)  # Same host, distinct endpoints are supported.
    robots[1]['robot_ip'] = 'nuc1'
    grippers[1]['port1'] = grippers[0]['port1']
    with pytest.raises(ValueError, match='separate serial ports'):
        validate_hardware_config(robots, grippers)


@pytest.mark.parametrize('entry', ['policy_client_pi.py', 'policy_client_pi_episodes.py'])
def test_sphere_correction_uses_legacy_offset(entry):
    solve = load_sphere_collision(entry)
    poses = np.array([[0., .80, 0, 0, 0, 0], [0., .01, 0, 0, 0, 0]])
    robots = [dict(sphere_radius=.1, sphere_center=[0, 0, 0])] * 2
    solve(poses, robots)
    np.testing.assert_allclose(poses[0, :3], [0, .75, 0])
    np.testing.assert_allclose(poses[1, :3], [0, .06, 0])
    np.testing.assert_allclose(poses[:, 3:], 0)


def test_temporal_average_handles_both_rotations_and_valid_zeros():
    actions = np.zeros((2, 14))
    actions[:, 5] = [0.1, 0.3]
    actions[:, 12] = [-0.2, -0.4]
    result = average_actions(actions, np.array([.5, .5]))
    np.testing.assert_allclose(result[0, [5, 12]], [.2, -.3], atol=1e-8)
    assert result[0, 0] == 0


def load_env():
    tree = ast.parse((ROOT / 'umi/real_world/bimanual_umi_env.py').read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BimanualUmiEnv')
    namespace = dict(np=np, time=time, math=math, pathlib=__import__('pathlib'), List=list,
                     validate_hardware_config=validate_hardware_config)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), 'bimanual_umi_env.py', 'exec'), namespace)
    return namespace['BimanualUmiEnv'], namespace


class FakeRobot:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.input_queue = types.SimpleNamespace(buffer_size=256, qsize=lambda: 0)
        self.calls = []
        self.is_ready = True
    def is_alive(self):
        return True
    def schedule_waypoint(self, **kwargs):
        self.calls.append(kwargs)
    def get_all_state(self):
        value = 0 if self.robot_ip == 'nuc0' else 1
        return dict(robot_timestamp=np.array([99., 100.]),
                    ActualTCPPose=np.tile([value, 0, 0, 0, 0, 0], (2, 1)))


class FakeGripper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.calls = []
        self.thread = types.SimpleNamespace(is_alive=lambda: True)
    def get_all_state(self):
        value = .01 if self.port1.endswith('0') else .07
        return dict(gripper_timestamp=np.array([99., 100.]), gripper_position=np.array([value, value]))
    def get_current_gripper_width(self):
        return self.get_all_state()['gripper_position'][-1]
    def move_gripper(self, width_m, **kwargs):
        self.calls.append(dict(width_m=width_m, **kwargs))


@pytest.mark.parametrize('same_nuc', [False, True])
def test_env_constructs_two_independent_workers_and_hands(tmp_path, monkeypatch, same_nuc):
    cls, ns = load_env()
    ns.update(FrankaInterpolationController=FakeRobot, RTDEInterpolationController=FakeRobot,
              GripperController=FakeGripper, optimal_row_cols=lambda **k: (4, 4, 2, 1),
              ReplayBuffer=types.SimpleNamespace(create_from_path=lambda **k: object()))
    monkeypatch.setitem(sys.modules, 'umi.real_world.mvs_camera_config',
                        types.SimpleNamespace(normalize_mvs_configs=lambda x, y: x))
    monkeypatch.setitem(sys.modules, 'umi.real_world.mvs_camera_service',
                        types.SimpleNamespace(IsolatedMultiMvsCamera=lambda **k: types.SimpleNamespace(n_cameras=2)))
    robots, grippers = configs()
    robots[1]['robot_port'] = 4243
    if same_nuc:
        robots[1]['robot_ip'] = robots[0]['robot_ip']
    robots[1]['controller_frequency'] = 999  # Not a supported configuration knob.
    env = cls(output_dir=tmp_path/'run', robots_config=robots, grippers_config=grippers,
              shm_manager=object(), enable_multi_cam_vis=False,
              cameras_config=[dict(width=4, height=4, fps=30)]*2)
    assert [r.robot_ip for r in env.robots] == ['nuc0', 'nuc0' if same_nuc else 'nuc1']
    assert [r.robot_port for r in env.robots] == [4242, 4243]
    assert [r.frequency for r in env.robots] == [200, 200]
    assert [g.port1 for g in env.grippers] == ['/dev/hand0', '/dev/hand1']
    assert env.grippers[0] is not env.grippers[1]
    with pytest.raises(AttributeError):
        _ = env.gripper_controller


def make_env():
    cls, ns = load_env()
    env = cls.__new__(cls)
    env.robots_config, env.grippers_config = configs()
    env.robots = [FakeRobot(**rc) for rc in env.robots_config]
    env.grippers = [FakeGripper(**gc) for gc in env.grippers_config]
    env.camera = types.SimpleNamespace(is_ready=True, n_cameras=2)
    env.action_accumulator = env.obs_accumulator = None
    return env, ns


def test_env_routes_both_pose_slices_and_grippers_with_shared_timeline():
    env, ns = make_env()
    ns['time'] = types.SimpleNamespace(time=lambda: 100.)
    actions = np.arange(42).reshape(3, 14).astype(float)
    env.exec_actions(actions, np.array([99.9, 101., 102.]), compensate_latency=True)
    for i in range(2):
        calls = env.robots[i].calls
        assert len(calls) == 2  # expired first row is discarded for both arms
        np.testing.assert_array_equal(calls[0]['pose'], actions[1, i*7:i*7+6])
        assert calls[0]['target_time'] == 101. - env.robots_config[i]['robot_action_latency']
        assert len(env.grippers[i].calls) == 1
        assert env.grippers[i].calls[0]['width_m'] == actions[1, i*7+6]




def test_get_obs_keeps_both_grippers_even_with_disjoint_camera_history():
    env, ns = make_env()
    from scipy.interpolate import interp1d
    ns['get_interp1d'] = lambda t, x: interp1d(t, x, axis=0, bounds_error=False, fill_value=(x[0], x[-1]))
    ns['PoseInterpolator'] = lambda t, x: lambda ts: np.tile(x[-1], (len(ts), 1))
    env.camera.get = lambda **k: {
        0: dict(timestamp=np.array([99.9, 100.]), color=np.zeros((2, 4, 4, 3), np.uint8)),
        1: dict(timestamp=np.array([99., 99.1]), color=np.ones((2, 4, 4, 3), np.uint8)),
    }
    env.camera_down_sample_steps = env.robot_down_sample_steps = env.gripper_down_sample_steps = 1
    env.camera_obs_horizon = env.robot_obs_horizon = env.gripper_obs_horizon = 2
    env.camera_capture_fps, env.frequency, env.last_camera_data = 30, 20, None
    obs = env.get_obs()
    assert obs['timestamp'][-1] == 99.1
    np.testing.assert_allclose(obs['robot0_gripper_width'], .01)
    np.testing.assert_allclose(obs['robot1_gripper_width'], .07)
    assert obs['robot0_eef_pos'][-1, 0] == 0
    assert obs['robot1_eef_pos'][-1, 0] == 1


@pytest.mark.parametrize('same_nuc', [False, True])
def test_episode_home_restarts_each_endpoint_preserving_tcp(same_nuc):
    tree = ast.parse((ROOT/'policy_client_pi_episodes.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'return_to_initial_joints')
    events = []
    class Worker(FakeRobot):
        def stop(self, wait):
            events.append(('stop', self.robot_ip, self.robot_port))
        def start(self, wait):
            events.append(('start', self.robot_ip, self.robot_port))
    robots = [Worker(robot_ip='nuc0' if same_nuc else f'nuc{i}', robot_port=4242+i, frequency=200,
                     receive_latency=0, tx_flange_tcp=np.eye(4), soft_real_time=False,
                     verbose=False, ring_buffer=types.SimpleNamespace(get_max_k=10),
                     Kx=np.ones(6), Kxd=np.ones(6)) for i in range(2)]
    env = types.SimpleNamespace(robots=robots)
    ns = dict(FrankaInterpolationController=Worker, time=types.SimpleNamespace(sleep=lambda t: None),
              wait_robot_ready=lambda r: events.append(('ready', r.robot_ip, r.robot_port)))
    exec(compile(ast.Module(body=[function], type_ignores=[]), 'episodes.py', 'exec'), ns)
    ns['return_to_initial_joints'](env, object(), [np.zeros(7), np.ones(7)], [4, 5])
    assert events == [(event, 'nuc0' if same_nuc else f'nuc{i}', 4242+i)
                      for i in range(2) for event in ('stop', 'start', 'ready')]
    assert [r.robot_port for r in env.robots] == [4242, 4243]
    np.testing.assert_array_equal(env.robots[1].joints_init, np.ones(7))






def test_policy_response_validated_before_any_hardware_submission():
    client = PolicyClient.__new__(PolicyClient)
    client.options, client.num_robots, client.prompt = contract(), 2, 'fold'
    client.timeout = 1
    sent = []
    client.packer = types.SimpleNamespace(pack=lambda payload: payload)
    client.codec = types.SimpleNamespace(unpackb=lambda message: message)
    client.ws = types.SimpleNamespace(send=sent.append, recv=lambda **kw: {'actions': np.zeros((32, 10))})
    with pytest.raises(ValueError, match='20'):
        client.call_server([np.zeros((4, 4, 3), np.uint8)] * 2, np.zeros(20))
    assert set(sent[0]) == {'right_image', 'left_image', 'reference', 'prompt'}


def test_observation_assembly_uses_both_camera_and_robot_indices(monkeypatch):
    # Input size already matches output, so isolate mapping from OpenCV availability.
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(INTER_AREA=3, resize=lambda image, *a, **kw: image.copy()))
    obs = {}
    for i in range(2):
        obs[f'camera{i}_rgb'] = np.full((2, 4, 4, 3), i*100, np.uint8)
        obs[f'robot{i}_eef_pos'] = np.array([[i, 0, 0]])
        obs[f'robot{i}_eef_rot_axis_angle'] = np.array([[0., 0, i*np.pi/2]])
        obs[f'robot{i}_gripper_width'] = np.array([[.03 + i*.03]])
    options = contract()
    images, state, poses, widths = build_observation(obs, 2, 4, [100/3, 100/3], options)
    assert images[0][0, 0, 0] == 0 and images[1][0, 0, 0] == 100
    assert state[0] == 0 and state[10] == 1  # right-first per-arm blocks
    np.testing.assert_allclose(state[[9, 19]], [1, 2])
    np.testing.assert_allclose(widths, [.03, .06])
    assert poses.shape == (2, 6)


@pytest.mark.parametrize('count', [1, 2])
@pytest.mark.parametrize('debug', [False, True])
@pytest.mark.parametrize('entry', ['policy_client_pi.py', 'policy_client_pi_episodes.py'])
def test_episode_main_one_complete_iteration_with_fake_hardware(tmp_path, monkeypatch, debug, entry, count):
    """Exercise request, decode, selection, saving and episode teardown together."""
    import atexit
    import click
    import contextlib
    import json
    import os
    import yaml
    from umi.real_world import bimanual_policy as bp
    robots, grippers = configs()
    robots, grippers = robots[:count], grippers[:count]
    for rc in robots:
        rc['height_threshold'] = -1
        rc['sphere_radius'] = 0
        rc['sphere_center'] = [0, 0, 0]
    config = dict(robots=robots, grippers=grippers,
                  cameras=[dict(serial='right'), dict(serial='left')][:count],
                  policy=dict(server_url='ws://unused', control_fps=20,
                              image_resolution=4, action_is_relative=False))
    cfg = tmp_path/'config.yaml'
    cfg.write_text(yaml.safe_dump(config))
    submissions, requests = [], []
    class Env:
        def __init__(self, **kwargs):
            self.video_dir = tmp_path/'videos'
            self.video_dir.mkdir()
            self.robots = [types.SimpleNamespace(get_state=lambda: {'ActualQ': np.zeros(7)}) for _ in range(count)]
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def get_obs(self):
            obs = dict(timestamp=np.array([time.time()-.02, time.time()-.01]))
            for i in range(count):
                obs[f'camera{i}_rgb'] = np.full((2, 4, 4, 3), i*100, np.uint8)
                obs[f'robot{i}_eef_pos'] = np.array([[float(i), 0, .3]])
                obs[f'robot{i}_eef_rot_axis_angle'] = np.zeros((1, 3))
                obs[f'robot{i}_gripper_width'] = np.array([[.03]])
            return obs
        def start_episode(self, start):
            pass
        def end_episode(self):
            pass
        def exec_actions(self, **kwargs):
            submissions.append(kwargs)
    class Client:
        metadata = {}
        def __init__(self, *args, **kwargs):
            assert kwargs['num_robots'] == count
        def close(self):
            pass
        def call_server(self, images, state):
            requests.append((images, state))
            return np.tile(state, (32, 1))
    tree = ast.parse((ROOT/entry).read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    fn.decorator_list = []
    keys = iter([{'c'}, set(), set(), set(), {'q'}])
    cv = types.SimpleNamespace(INTER_AREA=3, FONT_HERSHEY_SIMPLEX=0,
                               resize=lambda image, *a, **kw: image.copy(),
                               imwrite=lambda *a: True, imshow=lambda *a: None, pollKey=lambda: -1,
                               putText=lambda *a, **kw: None, setNumThreads=lambda *a: None)
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    raw_keys = iter([['c'], [], ['q']])
    counter = types.SimpleNamespace(get_press_events=lambda: next(raw_keys))
    ns = dict(vars(bp), os=os, sys=sys, time=time, json=json, yaml=yaml, click=click, cv2=cv,
              st=types.SimpleNamespace(Rotation=Rotation), atexit=atexit,
              SharedMemoryManager=lambda: contextlib.nullcontext(object()),
              KeystrokeCounter=lambda: contextlib.nullcontext(counter), KeyCode=lambda char: char, BimanualUmiEnv=Env,
              PolicyClient=Client, wait_robot_ready=lambda r: None,
              restore_initial_gripper_opening=lambda *a: None, precise_wait=lambda *a, **kw: None,
              read_episode_keys=lambda *a: next(keys), wait_pending_actions=lambda *a: None,
              finish_episode_recording=lambda *a: None, solve_table_collision=lambda **kw: None,
              solve_sphere_collision=load_sphere_collision(entry))
    stop_functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'episode_stop_reason']
    exec(compile(ast.Module(body=stop_functions + [fn], type_ignores=[]), entry, 'exec'), ns)
    # Avoid the fixed startup sleep while retaining real wall-clock timestamps.
    ns['time'] = types.SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda *a: None)
    ns['main'](None, None, None, str(tmp_path/'output'), str(cfg), None, 0, False,
               1, 60, 1, None, None, 2, False, False, False, 8, debug, True)
    assert len(requests) == 1 and len(requests[0][0]) == count
    assert requests[0][1].shape == (10 * count,)
    assert len(submissions) == (0 if debug else 1)
    if submissions:
        assert submissions[0]['actions'].shape == (2, 7 * count)
    captures = list((tmp_path/'output'/'policy_io').glob('run_*/*_execution.npz'))
    assert len(captures) == 1
    with np.load(captures[0]) as saved:
        assert saved['decoded_actions'].shape == (32, 7 * count)


def test_layout_is_implicit_and_default_order_is_right_left():
    options = contract()
    assert 'state_layout' not in options and 'action_layout' not in options
    assert 'arm_order' not in options
    client = PolicyClient.__new__(PolicyClient)
    client.options, client.num_robots, client.prompt = options, 2, 'fold'
    images = [np.full((4, 4, 3), 10, np.uint8), np.full((4, 4, 3), 20, np.uint8)]
    state = pack_state(np.arange(20).reshape(2, 10))
    payload = client.build_payload(images, state)
    assert payload['right_image'][0, 0, 0] == 10
    assert payload['left_image'][0, 0, 0] == 20
    np.testing.assert_array_equal(payload['reference'], np.arange(20))


@pytest.mark.parametrize('count', [1, 2])
def test_default_wire_contract_needs_only_hardware_count(count):
    options = policy_options({}, count)
    keys = ['observation/image_right'] if count == 1 else ['observation/image_right', 'observation/image_left']
    assert options == dict(image_keys=keys, state_key='current_tcp_pose9_gripper')
    client = PolicyClient.__new__(PolicyClient)
    client.options, client.num_robots, client.prompt = options, count, 'task'
    images = [np.full((4, 4, 3), i, np.uint8) for i in range(count)]
    payload = client.build_payload(images, np.arange(10 * count))
    assert set(payload) == set(keys) | {'prompt', 'current_tcp_pose9_gripper'}
    for i, key in enumerate(keys):
        assert payload[key][0, 0, 0] == i


@pytest.mark.parametrize('entry', ['policy_client_pi.py', 'policy_client_pi_episodes.py'])
def test_sphere_single_arm_zero_radii_and_coincident_centers(entry):
    solve = load_sphere_collision(entry)
    poses = np.zeros((2, 6))
    robots = [dict(sphere_radius=.1, sphere_center=[0, 0, 0])] * 2
    solve(poses[:1], robots[:1])
    solve(poses, [dict(sphere_radius=0, sphere_center=[0, 0, 0])] * 2)
    poses[0, 1] = .89
    with pytest.raises(ValueError, match='coincide'):
        solve(poses, robots)


def test_minimal_hardware_validation_keeps_count_and_gripper_mode_checks():
    robots, grippers = configs()
    with pytest.raises(ValueError, match='exactly one gripper'):
        validate_hardware_config(robots, grippers[:1])
    grippers[0]['mode'] = 'dual'
    with pytest.raises(ValueError, match='single-mode'):
        validate_hardware_config(robots, grippers)


@pytest.mark.parametrize('entry', ['policy_client_pi.py', 'policy_client_pi_episodes.py'])
def test_table_collision_corrects_lowest_rotated_finger(entry):
    solve_table_collision = load_collision(entry, 'solve_table_collision')
    pose = np.array([.1, .2, .01, 0., np.pi/2, 0.])
    before = pose.copy()
    solve_table_collision(pose, gripper_width=.06, height_threshold=.02)
    np.testing.assert_allclose(pose[2], .05)
    np.testing.assert_array_equal(pose[[0, 1, 3, 4, 5]], before[[0, 1, 3, 4, 5]])
    unchanged = pose.copy()
    solve_table_collision(pose, gripper_width=.06, height_threshold=.02)
    np.testing.assert_allclose(pose, unchanged)


@pytest.mark.parametrize('value', [np.nan, np.inf, -np.inf])
def test_nonfinite_translation_or_gripper_rejected(value):
    for index in (0, 9, 10, 19):
        actions = np.zeros((32, 20))
        actions[0, index] = value
        with pytest.raises(ValueError, match='non-finite'):
            unpack_actions(actions, 2)


@pytest.mark.parametrize('entry', ['policy_client_pi.py', 'policy_client_pi_episodes.py'])
@pytest.mark.parametrize('has_actions', [False, True])
def test_episode_recording_cleanup_handles_empty_and_nonempty_actions(entry, has_actions):
    fn = load_collision(entry, 'finish_episode_recording')
    events = []
    env = types.SimpleNamespace(
        obs_accumulator=types.SimpleNamespace(timestamps={'robot': np.array([1.])}),
        action_accumulator=types.SimpleNamespace(timestamps=np.array([1.]) if has_actions else np.array([])),
        camera=types.SimpleNamespace(stop_recording=lambda: events.append('stop')),
        end_episode=lambda: events.append('save'))
    fn(env)
    assert events == (['save'] if has_actions else ['stop'])
    if not has_actions:
        assert env.obs_accumulator is None and env.action_accumulator is None
        fn(env)
        assert events == ['stop']

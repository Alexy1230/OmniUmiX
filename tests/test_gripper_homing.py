"""Mechanical homing logic with a simulated clock/motor; never opens a port."""
import math
from types import SimpleNamespace
import pytest
from umi.real_world.gripper_homing import home_motor


class Motor:
    def __init__(self, velocity=0., valid=True, motion=False):
        self.velocity, self.position = velocity, -1.5
        self.refresh_ok = valid
        self.calls = []
        self.motion, self.steps, self.last_torque = motion, 0, 0.
    def refresh(self):
        self.calls.append('refresh')
        if self.motion and self.last_torque < 0:
            if self.steps < 8:
                self.position -= .01
                self.velocity = -.1
                self.steps += 1
            else:
                self.velocity = 0.

    def getPosition(self): return self.position
    def getVelocity(self): return self.velocity
    def set_torque_nm(self, value): self.calls.append(('torque',value)); self.last_torque = value
    def getTorque(self): return self.last_torque
    def set_zero_ram(self): self.calls.append('zero'); self.position = 0.


@pytest.fixture
def clock(monkeypatch):
    now = [0.]
    monkeypatch.setattr('umi.real_world.gripper_homing.time.monotonic',lambda:now[0])
    monkeypatch.setattr('umi.real_world.gripper_homing.time.sleep',lambda dt:now.__setitem__(0,now[0]+dt))
    return now


def test_mechanical_stop_then_zero(clock):
    motor=Motor(motion=True)
    home_motor(motor,dict(enabled=True))
    assert motor.calls.count('zero') == 1
    index=motor.calls.index('zero')
    assert ('torque',-0.05) in motor.calls[:index]
    assert motor.calls[index-1]==('torque',0)
    assert clock[0] >= 2.7


@pytest.mark.parametrize('velocity,valid',[(1.,True),(0.,False),(math.nan,True)])
def test_failure_stops_torque_and_never_sets_zero(clock,velocity,valid):
    motor=Motor(velocity,valid)
    with pytest.raises((TimeoutError,RuntimeError)):
        home_motor(motor,dict(enabled=True,timeout_s=.4))
    assert 'zero' not in motor.calls
    assert motor.calls[-1] == ('torque',0)


def test_bad_settings_do_not_command_motor(clock):
    motor=Motor()
    with pytest.raises(ValueError):home_motor(motor,dict(torque_nm=.05))
    assert motor.calls == []


def test_environment_homes_before_starting_arm_workers():
    import ast
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'umi/real_world/bimanual_umi_env.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BimanualUmiEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='start')
    scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    calls=[]
    def worker(name):return SimpleNamespace(start=lambda **kw:calls.append((name,kw)),stop=lambda **kw:None)
    env=SimpleNamespace(camera=worker('camera'),camera_startup_first=True,grippers=[worker('gripper')],
                        grippers_config=[dict(gripper_type='custom',homing=dict(enabled=True))],
                        robots=[worker('robot')],multi_cam_vis=None,start_wait=lambda:None)
    scope['start'](env)
    assert [c[0] for c in calls] == ['camera','gripper','robot']
    assert calls[1][1]['homing']['enabled'] is True


def test_homing_failure_cleans_up_and_never_starts_arm():
    import ast
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'umi/real_world/bimanual_umi_env.py'
    cls=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='BimanualUmiEnv')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='start')
    scope={};exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),scope)
    calls=[]
    def failure(**kw):raise TimeoutError('home failed')
    env=SimpleNamespace(camera=SimpleNamespace(start=lambda **kw:None,stop=lambda **kw:calls.append('camera_stop')),
        camera_startup_first=True,grippers=[SimpleNamespace(start=failure,stop=lambda:calls.append('gripper_stop'))],
        grippers_config=[dict(gripper_type='custom',homing=dict(enabled=True))],robots=[SimpleNamespace(start=lambda **kw:calls.append('robot_start'))],
        multi_cam_vis=None,start_wait=lambda:None)
    with pytest.raises(TimeoutError):scope['start'](env)
    assert calls==['gripper_stop','camera_stop']


def test_initial_opening_requires_fresh_reached_feedback(monkeypatch):
    import numpy as np
    from umi.real_world.client_deployment import wait_initial_gripper_opening
    t=[100.]
    monkeypatch.setattr('umi.real_world.client_deployment.time.time',lambda:t[0])
    monkeypatch.setattr('umi.real_world.client_deployment.time.monotonic',lambda:t[0])
    monkeypatch.setattr('umi.real_world.client_deployment.time.sleep',lambda dt:t.__setitem__(0,t[0]+dt))
    grip=SimpleNamespace(thread=SimpleNamespace(is_alive=lambda:True),
        get_all_state=lambda:{'gripper_timestamp':np.array([t[0]])},
        get_current_gripper_width=lambda:.055 if t[0]>.04+100 else .043)
    env=SimpleNamespace(grippers=[grip])
    wait_initial_gripper_opening(env,[dict(gripper_type='custom',initial_position=.055)])
    assert t[0]>=100.08


def test_initial_opening_timeout_blocks_warmup(monkeypatch):
    import numpy as np
    from umi.real_world.client_deployment import wait_initial_gripper_opening
    t=[100.]
    monkeypatch.setattr('umi.real_world.client_deployment.time.time',lambda:t[0])
    monkeypatch.setattr('umi.real_world.client_deployment.time.monotonic',lambda:t[0])
    monkeypatch.setattr('umi.real_world.client_deployment.time.sleep',lambda dt:t.__setitem__(0,t[0]+dt))
    grip=SimpleNamespace(thread=SimpleNamespace(is_alive=lambda:True),
        get_all_state=lambda:{'gripper_timestamp':np.array([t[0]])},get_current_gripper_width=lambda:.043)
    with pytest.raises(TimeoutError,match='not reached'):
        wait_initial_gripper_opening(SimpleNamespace(grippers=[grip]),[dict(gripper_type='custom',initial_position=.055)],timeout=.1)


def test_stationary_motor_is_not_accepted_as_mechanical_home(clock):
    motor=Motor()
    with pytest.raises(TimeoutError,match='no confirmed closing motion'):
        home_motor(motor,dict(enabled=True,timeout_s=.4))
    assert 'zero' not in motor.calls
    assert motor.calls[-1]==('torque',0)


def test_opening_motion_rejected_without_zero(clock):
    motor=Motor()
    def refresh():
        if motor.last_torque < 0:
            motor.position += .01
    motor.refresh=refresh
    with pytest.raises(RuntimeError,match='opening direction'):
        home_motor(motor,dict(enabled=True))
    assert 'zero' not in motor.calls and motor.calls[-1]==('torque',0)


@pytest.mark.parametrize('homing', [None, {}, {'enabled': False}, {'enabled': 'true'}])
def test_live_ftp1_rejects_config_that_would_skip_homing(homing):
    from umi.real_world.bimanual_policy import validate_hardware_config

    policy = {'protocol': 'ftp1_umi', 'tactile': {'enabled': True}}
    robots = [{'robot_ip': 'test-robot'}]
    gripper = {'gripper_type': 'custom', 'port': '/dev/fake'}
    if homing is not None:
        gripper['homing'] = homing

    # Other callers still support their existing hardware configuration.
    validate_hardware_config(robots, [gripper])
    with pytest.raises(ValueError, match=r'grippers\[0\].*homing.enabled: true'):
        validate_hardware_config(robots, [gripper], policy)


def test_sort_diff_profile_meets_live_homing_requirement():
    from pathlib import Path

    import yaml
    from umi.real_world.bimanual_policy import validate_hardware_config

    profile = Path(__file__).resolve().parents[1] / 'example/eval_ftp1_sort_diff.yaml'
    config = yaml.safe_load(profile.read_text())
    validate_hardware_config(config['robots'], config['grippers'], config['policy'])

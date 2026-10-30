"""Offline calibration tests: no serial ports, robot or camera are opened."""
import ast
import math
from pathlib import Path
import sys
import types
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from umi.real_world.bimanual_policy import configure_gripper_conversion, build_observation, decode_actions
from umi.real_world.gripper_calibration import width_to_rad_parameters


def calibration(lo=.01, hi=.08, rlo=0, rhi=3.5):
    return dict(min_abs_width_m=lo, max_abs_width_m=hi, min_rad=rlo, max_rad=rhi)


@pytest.mark.parametrize('bad', [{}, calibration(hi=.01), calibration(lo=-1), calibration(rhi=0), calibration(hi=float('nan')), calibration(rlo=None)])
def test_invalid_calibration_rejected(bad):
    with pytest.raises(ValueError):
        width_to_rad_parameters(bad)


@pytest.mark.parametrize('count', [1, 2])
def test_measured_state_and_action_roundtrip_per_arm(monkeypatch, count):
    monkeypatch.setitem(sys.modules, 'cv2', types.SimpleNamespace(INTER_AREA=3, resize=lambda x,*a,**k:x.copy()))
    configs = [calibration(), calibration(.02,.12,.5,4.5)][:count]
    options={}
    forward, inverse=configure_gripper_conversion([{'width_to_rad':c} for c in configs], 123, 456, options)
    widths=[.045,.095][:count]
    obs={}
    for i,w in enumerate(widths):
        obs[f'camera{i}_rgb']=np.zeros((1,4,4,3),dtype=np.uint8)
        obs[f'robot{i}_eef_pos']=np.array([[i,0.,0.]])
        obs[f'robot{i}_eef_rot_axis_angle']=np.zeros((1,3))
        obs[f'robot{i}_gripper_width']=np.array([[w]])
    _,state,poses,_=build_observation(obs,count,4,forward,options)
    np.testing.assert_allclose(state[9::10], [1.75,3.5][:count])
    result=decode_actions(np.tile(state,(32,1)),poses,inverse,False,options)
    np.testing.assert_allclose(result[:,6::7],np.tile(widths,(32,1)),atol=1e-8)
    # Only calibrated command widths are bounded, and pose values are untouched.
    raw=np.tile(state,(2,1));raw[0,9::10]=-100;raw[1,9::10]=100
    result=decode_actions(raw,poses,inverse,False,options)
    np.testing.assert_allclose(result[:,6::7], [[c['min_abs_width_m'] for c in configs],[c['max_abs_width_m'] for c in configs]])


def test_mixed_calibrated_and_legacy_and_initial_width():
    options={}
    f,r=configure_gripper_conversion([{'width_to_rad':calibration()},{}], [2,4], [.5,.25], options)
    np.testing.assert_allclose(f,[50,4]);np.testing.assert_allclose(r,[.02,.25])
    np.testing.assert_allclose(options['hardware_to_model_gripper_offset'],[-.5,0])
    with pytest.raises(ValueError,match='initial_position'):
        configure_gripper_conversion([{'width_to_rad':calibration(),'initial_position':0}], 1,1,{})


def test_motor_conversions_use_same_calibration_without_hardware():
    tree=ast.parse((ROOT/'umi/real_world/gripper_controller_1_14.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GripperController')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('_angle_to_width','_width_to_angle')]
    ns={'math':math};exec(compile(ast.Module(body=methods,type_ignores=[]),'motor_conversions','exec'),ns)
    motor=types.SimpleNamespace(width_calibration=width_to_rad_parameters(calibration()),MOTOR_PHYSICAL_LIMIT_M=.08,MM_PER_SINGLE_MOTOR_RADIAN=20)
    for angle,width in [(0,.01),(1.75,.045),(3.5,.08)]:
        assert ns['_angle_to_width'](motor,angle)==pytest.approx(width)
        assert ns['_width_to_angle'](motor,width)==pytest.approx(angle)
    assert ns['_width_to_angle'](motor,100)==pytest.approx(3.5)
    assert ns['_width_to_angle'](motor,-1)==pytest.approx(0)
    motor.width_calibration=None;motor.MM_PER_SINGLE_MOTOR_RADIAN=30
    assert ns['_angle_to_width'](motor,2)==pytest.approx(.06)
    assert ns['_width_to_angle'](motor,.06)==pytest.approx(2)

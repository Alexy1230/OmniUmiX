"""Hardware-free checks for episode references and paired request data."""
import numpy as np
from umi.real_world.proprio_policy_client import ProprioPolicyClient


def test_episode_padding_and_reset():
    client = ProprioPolicyClient.__new__(ProprioPolicyClient)
    client.num_robots, client.prompt = 1, 'test'
    client.options = {'image_keys': ['observation/image'], 'state_key': 'current_tcp_pose9_gripper'}
    client.reset_episode()
    state = np.array([0,0,0,1,0,0,0,1,0,1], dtype=np.float32)
    image = np.full((8,8,3), 20, dtype=np.uint8)
    older_image = np.full_like(image, 10)
    client.set_history(older_image, np.stack([state, state]), 0.9, 1.0)
    request = client.build_payload([image], state)
    np.testing.assert_array_equal(request['observation/image_previous'], image)
    later = state.copy(); later[0] = 0.1
    client.set_history(image, np.stack([state,later]), 1.0, 1.1)
    request = client.build_payload([older_image], later)
    np.testing.assert_array_equal(request['episode_start_tcp_pose9_gripper'], state)
    np.testing.assert_array_equal(request['tcp_pose9_gripper_history'], [state,later])
    np.testing.assert_array_equal(request['observation/image_previous'], image)
    client.reset_episode()
    client.set_history(image, np.stack([state,later]), 1.0, 1.1)
    request = client.build_payload([older_image], later)
    np.testing.assert_array_equal(request['episode_start_tcp_pose9_gripper'], later)
    np.testing.assert_array_equal(request['tcp_pose9_gripper_history'], [later,later])


def test_missing_history_rejected():
    client = ProprioPolicyClient.__new__(ProprioPolicyClient)
    client.num_robots, client.prompt = 1, 'test'
    client.options = {'image_keys': ['observation/image'], 'state_key': 'current_tcp_pose9_gripper'}
    client.reset_episode()
    try:
        client.build_payload([np.zeros((8,8,3),np.uint8)], np.zeros(10,np.float32))
    except ValueError:
        return
    raise AssertionError('Missing history must fail')


def test_dual_history_and_reference():
    client = ProprioPolicyClient.__new__(ProprioPolicyClient)
    client.num_robots, client.prompt = 2, 'test'
    client.options = {'image_keys': ['observation/image_right', 'observation/image_left'],
                      'state_key': 'current_tcp_pose9_gripper'}
    client.reset_episode()
    state = np.tile(np.array([0,0,0,1,0,0,0,1,0,1], np.float32), 2)
    right, left = [np.full((8,8,3), value, np.uint8) for value in (10,20)]
    client.set_history([right,left], np.stack([state,state]), 0.9, 1.0)
    client.build_payload([right,left], state)
    later = state.copy(); later[0]=0.1; later[10]=0.2
    client.set_history([right,left], np.stack([state,later]), 1.0, 1.1)
    request = client.build_payload([left,right], later)
    np.testing.assert_array_equal(request['observation/image_right_previous'],right)
    np.testing.assert_array_equal(request['observation/image_left_previous'],left)
    np.testing.assert_array_equal(request['tcp_pose9_gripper_history'],[state,later])
    np.testing.assert_array_equal(request['episode_start_tcp_pose9_gripper'],state)


def test_vision_payloads_have_no_proprio_history():
    from umi.real_world.bimanual_policy import PolicyClient, policy_options
    for count in (1,2):
        client = PolicyClient.__new__(PolicyClient)
        client.num_robots, client.prompt = count, 'test'
        client.options = policy_options({},count)
        payload = client.build_payload([np.zeros((8,8,3),np.uint8)]*count,np.zeros(10*count,np.float32))
        assert 'tcp_pose9_gripper_history' not in payload
        assert len([k for k in payload if k.startswith('observation/image')]) == count


def test_single_server_rejected_for_dual_client():
    client = ProprioPolicyClient.__new__(ProprioPolicyClient)
    client.num_robots = 2
    client.options = {'image_keys': ['observation/image_right','observation/image_left'],
                      'state_key': 'current_tcp_pose9_gripper'}
    client.metadata = dict(proprioception=True,image_history_horizon=2,is_relative=False,state_dim=32)
    try:
        client.validate_contract()
    except ValueError:
        pass
    else:
        raise AssertionError('Single-arm model accepted for dual-arm deployment')
    client.metadata.update(state_dim=64,num_robots=2,arm_order=['right','left'])
    client.validate_contract()

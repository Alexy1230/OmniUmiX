"""FTP1 tactile/force protocol regression tests; never open real devices."""
import json
import io
from types import SimpleNamespace
import numpy as np
import pytest
from umi.real_world.bimanual_policy import PolicyClient, policy_options
from umi.real_world.proprio_policy_client import ProprioPolicyClient
from umi.real_world.xense_capture import SensorPair, validate_config
from umi.common import openpi_msgpack_numpy as codec


def config(tmp_path, *, record_force=True):
    for serial in ('LEFT', 'RIGHT'):
        (tmp_path/f'runtime_{serial}').write_bytes(serial.encode())
    return dict(left=dict(serial='LEFT', runtime_path=str(tmp_path/'runtime_LEFT')),
                right=dict(serial='RIGHT', runtime_path=str(tmp_path/'runtime_RIGHT')),
                max_stale_s=1, runtime_mode='saved', record_force=record_force)


class FakeSolver:
    released = []

    def __init__(self, path):
        self.side = 'LEFT' if str(path).endswith('LEFT') else 'RIGHT'

    def selectSensorInfo(self, *outputs, rectify_image):
        assert rectify_image.shape == (700, 400, 3)
        # Preserve channel-dependent values so RGB/BGR mistakes are visible.
        image = rectify_image.copy()
        return (image, np.arange(6)) if len(outputs) == 2 else image

    def release(self):
        self.released.append(self.side)


class FakeSensor:
    OutputType = SimpleNamespace(Rectify=0, Difference=1, ForceResultant=2, TimeStamp=3)
    released, calibrated = [], []
    calls = []

    @staticmethod
    def scanSerialNumber():
        return {'LEFT': 0, 'RIGHT': 1}

    @classmethod
    def create(cls, **kwargs):
        assert kwargs['infer_mode'] == 'normal'
        return cls(kwargs['cam_id'])

    @staticmethod
    def createSolver(path):
        return FakeSolver(path)

    def __init__(self, cam_id):
        self.cam_id = cam_id

    def selectSensorInfo(self, *outputs):
        assert outputs == (0, 3), 'Live Sensor must acquire Rectify, not Difference'
        self.calls.append((self.cam_id, 'read'))
        return np.full((700, 400, 3), [10+self.cam_id,20,30], np.uint8), 1.0

    def calibrateSensor(self):
        self.calibrated.append(self.cam_id)
        self.calls.append((self.cam_id, 'calibrate'))

    def exportRuntimeConfig(self, save_dir, binary):
        from pathlib import Path
        assert binary is False
        serial = ('LEFT','RIGHT')[self.cam_id]
        (Path(save_dir)/f'runtime_{serial}').write_bytes(serial.encode())
        self.calls.append((self.cam_id, 'export'))

    def release(self):
        self.released.append(self.cam_id)


def client(cls=PolicyClient, force=True):
    obj = cls.__new__(cls)
    obj.options = policy_options({'protocol':'ftp1_umi', 'action_is_relative':False}, 1)
    obj.num_robots, obj.prompt, obj.timeout = 1, 'grasp', 1
    obj.tactile = SimpleNamespace(read=lambda: {
        side: dict(image_rgb=np.full((7,4,3), i,np.uint8), force_resultant=np.arange(6),
                   sensor_timestamp=1., received_at=2.)
        for i,side in enumerate(('left','right'))})
    obj.metadata = {'predict_finger_force':force}
    obj.packer, obj.codec = codec.Packer(), codec
    obj.response_log = io.StringIO()
    action = np.tile([.1,.2,.3,1,0,0,0,1,0,2.0], (32,1)).astype(np.float32)
    response = dict(actions=action)
    if force:
        response['finger_force'] = np.arange(32*2*3,dtype=np.float32).reshape(32,2,3)
    obj.ws = SimpleNamespace(send=lambda payload: setattr(obj,'sent',codec.unpackb(payload)),
                             recv=lambda **kwargs: codec.packb(response))
    return obj, response


def test_difference_conversion_and_release(tmp_path):
    FakeSensor.released = []
    pair = SensorPair(config(tmp_path), FakeSensor)
    try:
        data = pair.read()
        np.testing.assert_array_equal(data['left']['image_rgb'][0,0], [30,20,10])
        np.testing.assert_array_equal(data['right']['image_rgb'][0,0], [30,20,11])
        np.testing.assert_array_equal(data['left']['force_resultant'], np.arange(6))
    finally:
        pair.close()
    assert FakeSensor.released == [0,1]


def test_no_placeholder_or_duplicate_serials():
    with pytest.raises(ValueError,match='serial'):
        validate_config({'left':{'serial':''},'right':{'serial':'R'}})
    with pytest.raises(ValueError,match='distinct'):
        validate_config({'left':{'serial':'R'},'right':{'serial':'R'}})


def test_stale_frame_rejected(monkeypatch,tmp_path):
    pair = SensorPair(config(tmp_path), FakeSensor)
    monkeypatch.setattr('umi.real_world.xense_capture.time.monotonic', lambda: pair.previous['left'][1]+2)
    try:
        with pytest.raises(RuntimeError, match='stopped advancing'):
            pair.read()
    finally:
        pair.close()


@pytest.mark.parametrize('force',[False,True])
def test_tactile_wire_and_force_reply(force):
    obj, response = client(force=force)
    state = response['actions'][0]
    actual = obj.call_server([np.zeros((240,240,3),np.uint8)], state)
    np.testing.assert_array_equal(actual,response['actions'])
    assert set(obj.sent) == {'observation/image','current_tcp_pose9_gripper','prompt',
                            'observation/tactile_left_difference','observation/tactile_right_difference'}
    assert obj.sent['observation/tactile_right_difference'][0,0,0] == 1
    if force:
        np.testing.assert_array_equal(obj.last_finger_force,response['finger_force'])
    record = json.loads(obj.response_log.getvalue())
    assert record['request_id'] == 1 and len(record['actions']) == 32
    assert record['measured']['left']['force_resultant'] == list(range(6))


@pytest.mark.parametrize('shape',[(32,3,2),(31,2,3),(32,6)])
def test_wrong_force_shape_rejected(shape):
    obj, response = client()
    response['finger_force'] = np.zeros(shape,np.float32)
    with pytest.raises(ValueError,match='finger_force'):
        obj.call_server([np.zeros((4,4,3),np.uint8)], response['actions'][0])


def test_force_checkpoint_missing_output_rejected():
    obj, response = client()
    del response['finger_force']
    with pytest.raises(ValueError,match='omitted'):
        obj.call_server([np.zeros((4,4,3),np.uint8)], response['actions'][0])


def test_proprio_entry_uses_ftp1_current_state_contract():
    obj, response = client(ProprioPolicyClient)
    obj.protocol = 'ftp1_umi'
    obj.validate_contract()
    request = obj.build_payload([np.zeros((4,4,3),np.uint8)], response['actions'][0])
    assert 'tcp_pose9_gripper_history' not in request
    assert 'observation/tactile_left_difference' in request


def test_relative_or_unsupported_dual_checkpoint_rejected():
    with pytest.raises(ValueError,match='absolute'):
        policy_options({'protocol':'ftp1_umi','action_is_relative':True},1)
    with pytest.raises(ValueError,match='one RIGHT arm'):
        policy_options({'protocol':'ftp1_umi','action_is_relative':False},2)


def test_real_constructor_handshake_and_cleanup(monkeypatch, tmp_path):
    import websockets.sync.client
    import umi.real_world.xense_capture as capture_module
    template, response = client()
    messages = iter([codec.packb(dict(protocol='ftp1_umi',action_type='tcp_pose9_gripper',is_relative=False,num_robots=1,
                                      use_tactile_input=True,predict_finger_force=True)), codec.packb(response)])
    closed = []
    transport = SimpleNamespace(recv=lambda **kwargs:next(messages),send=lambda message:None,
                                close=lambda:closed.append('websocket'))
    monkeypatch.setattr(websockets.sync.client,'connect',lambda *a,**k:transport)
    provider = SimpleNamespace(start=lambda:provider,read=template.tactile.read,
                               close=lambda:closed.append('xense'))
    monkeypatch.setattr(capture_module,'XenseCapture',lambda cfg:provider)
    cfg = dict(protocol='ftp1_umi',action_is_relative=False,tactile={'enabled':True},capture_dir=str(tmp_path))
    obj = PolicyClient('ws://unused','grasp',policy_config=cfg)
    try:
        np.testing.assert_array_equal(obj.call_server([np.zeros((4,4,3),np.uint8)],response['actions'][0]),response['actions'])
    finally:
        obj.close()
        obj.close()
    assert closed == ['xense','websocket']
    assert (tmp_path/'policy_responses.jsonl').is_file()


@pytest.mark.parametrize('devices_present',[False,True])
def test_isolated_process_startup_and_shutdown(tmp_path,monkeypatch,devices_present):
    import inspect
    import os
    from umi.real_world.xense_capture import XenseCapture
    source = 'from types import SimpleNamespace\nimport numpy as np\n' + inspect.getsource(FakeSolver) + '\n' + inspect.getsource(FakeSensor).replace('class FakeSensor:', 'class Sensor:')
    if not devices_present:
        source = source.replace("return {'LEFT': 0, 'RIGHT': 1}", 'return {}')
    (tmp_path/'xensesdk.py').write_text(source)
    (tmp_path/'onnxruntime.py').write_text('def preload_dlls(**kwargs): pass\n')
    monkeypatch.setenv('PYTHONPATH',str(tmp_path)+os.pathsep+os.environ.get('PYTHONPATH',''))
    options = config(tmp_path)
    options['startup_timeout_s'] = 10
    if os.environ.get('UMI_TEST_XENSE_PYTHON'):
        options['python'] = os.environ['UMI_TEST_XENSE_PYTHON']
    capture = XenseCapture(options)
    if devices_present:
        capture.start()
        worker = capture.process
        try:
            sample = capture.read()
            np.testing.assert_array_equal(sample['left']['image_rgb'][0,0],[30,20,10])
        finally:
            capture.close()
        assert worker.poll() is not None
    else:
        with pytest.raises(RuntimeError,match='absent; available='):
            capture.start()
    assert capture.directory is None and capture.process is None


@pytest.mark.parametrize('legacy',[False,True])
def test_sort_diff_works_with_original_three_key_server(monkeypatch,tmp_path,legacy):
    import websockets.sync.client
    import umi.real_world.xense_capture as module
    template, response = client(force=False)
    metadata = dict(action_type='tcp_pose9_gripper',is_relative=False,action_horizon=32)
    if not legacy:
        metadata.update(protocol='ftp1_umi',use_tactile_input=True,predict_finger_force=False,num_robots=1)
    messages = iter([codec.packb(metadata),codec.packb(response)])
    sent = []
    transport = SimpleNamespace(recv=lambda **kwargs:next(messages),send=lambda msg:sent.append(codec.unpackb(msg)),close=lambda:None)
    monkeypatch.setattr(websockets.sync.client,'connect',lambda *a,**k:transport)
    provider = SimpleNamespace(start=lambda:provider,read=template.tactile.read,close=lambda:None)
    monkeypatch.setattr(module,'XenseCapture',lambda cfg:provider)
    options = dict(protocol='ftp1_umi',action_is_relative=False,tactile={'enabled':True})
    obj = PolicyClient('ws://unused','grasp',policy_config=options)
    try:
        actual = obj.call_server([np.zeros((240,240,3),np.uint8)],response['actions'][0])
        np.testing.assert_array_equal(actual,response['actions'])
        assert obj.last_finger_force is None
        assert sent[0]['observation/tactile_left_difference'].shape == (224,224,3)
        assert set(sent[0]) == {'prompt','current_tcp_pose9_gripper','observation/image',
                              'observation/tactile_left_difference','observation/tactile_right_difference'}
    finally:
        obj.close()


def test_sort_does_not_require_force(tmp_path):
    pair = SensorPair(config(tmp_path, record_force=False),FakeSensor)
    try:
        sample = pair.read()
        assert 'force_resultant' not in sample['left']
        np.testing.assert_array_equal(sample['left']['image_rgb'][0,0],[30,20,10])
        assert len(sample['left']['runtime']['sha256']) == 64
    finally:
        pair.close()


def test_warmup_then_calibration_export_then_solver_read(tmp_path):
    options = config(tmp_path)
    options.update(runtime_mode='calibrate',runtime_directory=str(tmp_path/'export'))
    FakeSensor.calls, FakeSensor.calibrated = [],[]
    pair = SensorPair(options,FakeSensor)
    try:
        for i in (0,1):
            reads = [idx for idx,call in enumerate(FakeSensor.calls) if call == (i,'read')]
            assert reads[0] < FakeSensor.calls.index((i,'calibrate')) < FakeSensor.calls.index((i,'export')) < reads[-1]
        manifest = list((tmp_path/'export').glob('xrs_*/manifest.json'))
        assert len(manifest)==1
        assert set(json.loads(manifest[0].read_text())['sensors'])=={'right_hand_left_finger','right_hand_right_finger'}
    finally:
        pair.close()


def test_calibration_requires_confirmation_without_opening_sdk(monkeypatch,tmp_path):
    from umi.real_world.xense_capture import XenseCapture
    options = config(tmp_path)
    options['runtime_mode'] = 'calibrate'
    monkeypatch.setattr('builtins.input',lambda prompt:'cancel')
    capture = XenseCapture(options)
    with pytest.raises(RuntimeError,match='cancelled'):
        capture.start()
    assert capture.process is None


@pytest.mark.parametrize('text',['calibrate','CALIBRATE',' Calibrate ','ｃａｌｉｂｒａｔｅ'])
def test_calibration_confirmation_accepts_normalized_command(monkeypatch,text):
    from umi.real_world.xense_capture import confirm_unloaded_calibration
    monkeypatch.setattr('builtins.input',lambda prompt:text)
    confirm_unloaded_calibration()


def test_empty_line_and_typo_retry_instead_of_cancel(monkeypatch,capsys):
    from umi.real_world.xense_capture import confirm_unloaded_calibration
    entries=iter(['','c','calibiration','calibrate'])
    reads=[]
    def read(prompt):
        value=next(entries);reads.append(value);return value
    monkeypatch.setattr('builtins.input',read)
    confirm_unloaded_calibration()
    assert len(reads)==4
    assert capsys.readouterr().out.count('Still waiting')==3


@pytest.mark.parametrize('text',['q','QUIT',' cancel '])
def test_only_explicit_cancel_exits_confirmation(monkeypatch,text):
    from umi.real_world.xense_capture import confirm_unloaded_calibration
    monkeypatch.setattr('builtins.input',lambda prompt:text)
    with pytest.raises(RuntimeError,match='cancelled by user'):
        confirm_unloaded_calibration()


def test_noninteractive_confirmation_reports_terminal_requirement(monkeypatch):
    from umi.real_world.xense_capture import confirm_unloaded_calibration
    def eof(prompt): raise EOFError
    monkeypatch.setattr('builtins.input',eof)
    with pytest.raises(RuntimeError,match='interactive terminal'):
        confirm_unloaded_calibration()


def test_main_client_inference_saves_real_request_images(monkeypatch,tmp_path,capsys):
    import cv2
    import websockets.sync.client
    import umi.real_world.xense_capture as module
    scene=np.full((240,240,3),[20,40,60],np.uint8)
    left=np.full((700,400,3),[10,30,90],np.uint8)
    right=np.full((700,400,3),[90,30,10],np.uint8)
    state=np.array([.3,.1,.2,1,0,0,0,1,0,1.5],np.float32)
    messages=iter([codec.packb(dict(action_type='tcp_pose9_gripper',is_relative=False,action_horizon=32)),
                   codec.packb({'actions':np.tile(state,(32,1))}),codec.packb({'actions':np.tile(state,(32,1))})])
    sent=[]
    transport=SimpleNamespace(recv=lambda **kwargs:next(messages),send=lambda msg:sent.append(codec.unpackb(msg)),close=lambda:None)
    monkeypatch.setattr(websockets.sync.client,'connect',lambda *a,**k:transport)
    provider=SimpleNamespace(start=lambda:provider,read=lambda:{
        side:dict(image_rgb=rgb,sensor_timestamp=1.,runtime={'sha256':'mock'})
        for side,rgb in [('left',left),('right',right)]},close=lambda:None)
    monkeypatch.setattr(module,'XenseCapture',lambda config:provider)
    obj=PolicyClient('ws://unused','sort',policy_config=dict(protocol='ftp1_umi',action_is_relative=False,
                     tactile={'enabled':True},save_tactile_images=True,capture_dir=str(tmp_path)))
    for phase in ['warmup','policy']:
        obj.capture_phase=phase
        obj.call_server([scene],state)
    directory=obj.input_recorder.directory
    obj.close()
    for index,phase in enumerate(['warmup','policy'],1):
        frame=directory/f'request_{index:06d}'
        metadata=json.loads((frame/'metadata.json').read_text())
        assert metadata['phase']==phase
        for side,rgb in [('left',left),('right',right)]:
            np.testing.assert_array_equal(cv2.imread(str(frame/f'{side}_difference_rgb.png'))[...,::-1],rgb)
            np.testing.assert_array_equal(cv2.imread(str(frame/f'tactile_{side}_difference_rgb.png'))[...,::-1],
                                          sent[index-1][f'observation/tactile_{side}_difference'])
        np.testing.assert_array_equal(np.load(frame/'state.npy'),state)
    output=capsys.readouterr().out
    assert 'Enabled for this CLIENT run' in output
    assert 'Saved first warmup' in output and 'Saved first policy' in output

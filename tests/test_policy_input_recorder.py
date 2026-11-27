import json
import numpy as np
import pytest
from umi.real_world.policy_input_recorder import PolicyInputRecorder


def test_lossless_raw_and_exact_model_inputs(tmp_path):
    import cv2
    raw=np.full((700,400,3),[20,50,190],np.uint8)
    model=cv2.resize(raw,(224,224))
    payload={'observation/image':np.zeros((240,240,3),np.uint8),
             'current_tcp_pose9_gripper':np.arange(10,dtype=np.float32),'prompt':'sort',
             'observation/tactile_left_difference':model}
    tactile={'left':dict(image_rgb=raw,sensor_timestamp=1.,runtime={'sha256':'test'})}
    recorder=PolicyInputRecorder(tmp_path)
    recorder.submit(1,payload,tactile,{},phase='warmup',observation_timestamp=.9)
    recorder.close()
    frame=recorder.directory/'request_000001'
    np.testing.assert_array_equal(cv2.imread(str(frame/'left_difference_rgb.png'))[...,::-1],raw)
    np.testing.assert_array_equal(cv2.imread(str(frame/'tactile_left_difference_rgb.png'))[...,::-1],model)
    np.testing.assert_array_equal(np.load(frame/'state.npy'),payload['current_tcp_pose9_gripper'])
    metadata=json.loads((frame/'metadata.json').read_text())
    assert metadata['phase']=='warmup' and metadata['observation_timestamp']==.9
    assert metadata['sensors']['left']['runtime']['sha256']=='test'


def test_capture_surfaces_write_failure(tmp_path,monkeypatch):
    import cv2
    monkeypatch.setattr(cv2,'imwrite',lambda *a,**kw:False)
    recorder=PolicyInputRecorder(tmp_path)
    recorder.submit(1,{'observation/image':np.zeros((2,2,3),np.uint8),'current_tcp_pose9_gripper':np.zeros(10)},None,{})
    with pytest.raises(IOError):recorder.close()

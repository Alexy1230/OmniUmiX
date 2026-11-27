import pytest
from umi.real_world.bimanual_policy import policy_options

@pytest.mark.parametrize("config", [
    {"image_keys": ["prompt"]},
    {"image_keys": ["current_tcp_pose9_gripper"]},
    {"state_key": "prompt"},
    {"image_keys": ["right", "state"], "state_key": "state"},
])
def test_request_fields_cannot_overwrite_each_other(config):
    count = len(config.get("image_keys", [None]))
    with pytest.raises(ValueError, match="distinct"):
        policy_options(config, count)

from openpi.policies import umi_bimanual_policy
from openpi.training import config
from openpi.training import data_loader
from openpi import transforms_umi_bimanual


def test_bimanual_config_matches_client_contract():
    train = config.get_config("pi05_umi_bimanual")
    data = train.data.create(train.assets_dirs, train.model)
    assert train.model.action_dim == 32 and train.model.action_horizon == 32
    assert not train.model.discrete_state_input
    assert data.normalization_mode == "umi_action20"
    assert data.action_sequence_keys == ("action",)
    assert isinstance(data.repack_transforms.inputs[0], umi_bimanual_policy.UmiBimanualRepack)
    assert isinstance(data_loader._make_normalization_transform(data, {}), transforms_umi_bimanual.UMIAction20Normalize)
    assert train.policy_metadata["arm_order"] == ["right", "left"]
    assert train.policy_metadata["is_relative"] is False


def test_single_arm_config_unchanged():
    train = config.get_config("pi05_umi_lift_bottle")
    data = train.data.create(train.assets_dirs, train.model)
    assert data.normalization_mode == "umi_action10"

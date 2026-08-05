from alpha.config import Config, set_global_seed


def test_config_output_root_default():
    assert Config.OUTPUT_ROOT == "outputs"


def test_config_registry_file_preserved():
    assert Config.REGISTRY_FILE == "registry_academic.json"


def test_set_global_seed_runs():
    set_global_seed(42)


def test_config_has_reward_config():
    assert 'zs500' in Config.REWARD_CONFIG
    assert 'hs300' in Config.REWARD_CONFIG

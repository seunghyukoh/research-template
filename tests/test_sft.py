"""Tests for the SFT package."""


def test_sft_imports():
    from sft import SFTTrainer
    from sft.config import SFTConfig

    assert SFTTrainer is not None
    assert SFTConfig is not None


def test_sft_config_defaults():
    from sft.config import SFTConfig

    config = SFTConfig()
    assert config.model.model_name == "gpt2"
    assert config.model.use_lora is True
    assert config.training.num_train_epochs == 3

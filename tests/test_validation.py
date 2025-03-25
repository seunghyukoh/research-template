"""Tests for validation utilities."""

import pytest

from src.packages.utils.exceptions import ConfigurationError
from src.packages.utils.validation import (
    validate_dataset_config,
    validate_model_config,
    validate_training_config,
)


def test_validate_model_config_success():
    """Test successful model config validation."""
    config = {
        "model_path": "gpt2",
        "tokenizer_path": "gpt2",
        "use_lora": True,
        "lora_rank": 8,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    validate_model_config(config)  # Should not raise


def test_validate_model_config_missing_required():
    """Test model config validation with missing required fields."""
    config = {"model_path": "gpt2"}  # Missing tokenizer_path
    with pytest.raises(ConfigurationError, match="Missing required field"):
        validate_model_config(config)


def test_validate_model_config_invalid_lora():
    """Test model config validation with invalid LoRA settings."""
    config = {
        "model_path": "gpt2",
        "tokenizer_path": "gpt2",
        "use_lora": True,
        "lora_rank": -1,  # Invalid value
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    with pytest.raises(ConfigurationError, match="Invalid value for lora_rank"):
        validate_model_config(config)


def test_validate_training_config_success():
    """Test successful training config validation."""
    config = {
        "output_dir": "output",
        "num_train_epochs": 3,
        "per_device_train_batch_size": 8,
    }
    validate_training_config(config)  # Should not raise


def test_validate_training_config_missing_required():
    """Test training config validation with missing required fields."""
    config = {"output_dir": "output"}  # Missing other required fields
    with pytest.raises(ConfigurationError, match="Missing required field"):
        validate_training_config(config)


def test_validate_training_config_invalid_values():
    """Test training config validation with invalid values."""
    config = {
        "output_dir": "output",
        "num_train_epochs": 0,  # Invalid value
        "per_device_train_batch_size": 8,
    }
    with pytest.raises(ConfigurationError, match="num_train_epochs must be positive"):
        validate_training_config(config)


def test_validate_dataset_config_success():
    """Test successful dataset config validation."""
    config = {
        "dataset_name": "custom",
        "dataset_path": "data/custom",
        "max_train_samples": 1000,
        "max_validation_samples": 100,
        "shuffle_seed": 42,
    }
    validate_dataset_config(config, allowed_datasets=["custom"])  # Should not raise


def test_validate_dataset_config_missing_required():
    """Test dataset config validation with missing required fields."""
    config = {"dataset_name": "custom"}  # Missing dataset_path
    with pytest.raises(ConfigurationError, match="dataset_path must be specified"):
        validate_dataset_config(config)


def test_validate_dataset_config_invalid_dataset():
    """Test dataset config validation with invalid dataset name."""
    config = {
        "dataset_name": "invalid",
        "dataset_path": "data/custom",
        "shuffle_seed": 42,
    }
    with pytest.raises(
        ConfigurationError,
        match="dataset_name must be one of \\['custom'\\], got invalid",
    ):
        validate_dataset_config(config, allowed_datasets=["custom"])

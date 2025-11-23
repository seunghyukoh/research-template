"""Unit tests for experiment runner functionality."""

import math
import os

# Import functions to test
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add parent directory to path to import run_experiment
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Skip all tests if dependencies are not available
try:
    from run_experiment import (
        calculate_num_gpus_needed,
        check_run_status_from_cache,
        convert_config_to_hydra_args,
        detect_gpu_memory_per_device,
        dict_to_hydra_args,
        fetch_wandb_run_status_cache,
    )
except ImportError as e:
    pytestmark = pytest.mark.skip(reason=f"Dependencies not available: {e}")


class TestDictToHydraArgs:
    """Test dict_to_hydra_args function."""

    def test_simple_dict(self):
        """Test conversion of simple dictionary."""
        args = {"learning_rate": 0.001, "batch_size": 32}
        result = dict_to_hydra_args(args)
        assert "learning_rate=0.001" in result
        assert "batch_size=32" in result

    def test_nested_dict(self):
        """Test conversion of nested dictionary."""
        args = {"training": {"learning_rate": 0.001, "batch_size": 32}}
        result = dict_to_hydra_args(args)
        assert "training.learning_rate=0.001" in result
        assert "training.batch_size=32" in result

    def test_list_values(self):
        """Test conversion of list values."""
        args = {"tags": ["experiment", "test"]}
        result = dict_to_hydra_args(args)
        assert "tags=['experiment', 'test']" in result

    def test_custom_args_prefix(self):
        """Test custom args with + prefix."""
        args = {"tags": ["test"]}
        result = dict_to_hydra_args(args, is_custom_args=True)
        assert any(arg.startswith("+") for arg in result)

    def test_deeply_nested_dict(self):
        """Test deeply nested dictionary."""
        args = {"a": {"b": {"c": "value"}}}
        result = dict_to_hydra_args(args)
        assert "a.b.c=value" in result

    def test_empty_dict(self):
        """Test empty dictionary."""
        args = {}
        result = dict_to_hydra_args(args)
        assert result == []


class TestConvertConfigToHydraArgs:
    """Test convert_config_to_hydra_args function."""

    def test_none_config(self):
        """Test with None config."""
        result = convert_config_to_hydra_args(None)
        assert result == []

    @patch("run_experiment.OmegaConf")
    def test_omegaconf_conversion(self, mock_omegaconf):
        """Test OmegaConf conversion."""
        mock_config = Mock()
        mock_omegaconf.to_container.return_value = {"key": "value"}

        result = convert_config_to_hydra_args(mock_config)
        mock_omegaconf.to_container.assert_called_once()
        assert "key=value" in result


class TestCalculateNumGpusNeeded:
    """Test calculate_num_gpus_needed function."""

    def test_exact_fit_single_gpu(self):
        """Test when memory requirement exactly fits one GPU."""
        gpu_groups = {40: [0, 1, 2, 3]}
        num_gpus, memory = calculate_num_gpus_needed(40, gpu_groups)
        assert num_gpus == 1
        assert memory == 40

    def test_requires_multiple_gpus(self):
        """Test when memory requirement needs multiple GPUs."""
        gpu_groups = {40: [0, 1, 2, 3]}
        num_gpus, memory = calculate_num_gpus_needed(80, gpu_groups)
        assert num_gpus == 2
        assert memory == 40

    def test_requires_partial_gpu(self):
        """Test when memory requirement is between GPU capacities."""
        gpu_groups = {40: [0, 1, 2, 3]}
        num_gpus, memory = calculate_num_gpus_needed(50, gpu_groups)
        assert num_gpus == 2  # ceil(50/40) = 2
        assert memory == 40

    def test_mixed_gpu_types_prefer_smaller(self):
        """Test with mixed GPU types, should prefer smaller sufficient GPU."""
        gpu_groups = {40: [0, 1], 80: [2, 3]}
        num_gpus, memory = calculate_num_gpus_needed(70, gpu_groups)
        # Should use 80GB GPU since 40GB * 2 = 80 but we have 80GB available
        assert num_gpus == 1
        assert memory == 80

    def test_insufficient_gpus_raises_error(self):
        """Test error when not enough GPUs available."""
        gpu_groups = {40: [0, 1]}  # Only 2 GPUs with 40GB each
        with pytest.raises(ValueError, match="Cannot satisfy memory requirement"):
            calculate_num_gpus_needed(200, gpu_groups)  # Need 5 GPUs

    def test_no_gpus_raises_error(self):
        """Test error when no GPUs available."""
        gpu_groups = {}
        with pytest.raises(ValueError, match="No GPUs available"):
            calculate_num_gpus_needed(40, gpu_groups)

    def test_small_memory_requirement(self):
        """Test with very small memory requirement."""
        gpu_groups = {40: [0, 1, 2, 3]}
        num_gpus, memory = calculate_num_gpus_needed(10, gpu_groups)
        assert num_gpus == 1  # ceil(10/40) = 1
        assert memory == 40


class TestCheckRunStatusFromCache:
    """Test check_run_status_from_cache function."""

    def test_task_exists_in_cache(self):
        """Test when task exists in cache."""
        cache = {"task_123": (True, "finished")}
        exists, status = check_run_status_from_cache("task_123", cache)
        assert exists is True
        assert status == "finished"

    def test_task_not_in_cache(self):
        """Test when task doesn't exist in cache."""
        cache = {"task_123": (True, "finished")}
        exists, status = check_run_status_from_cache("task_456", cache)
        assert exists is False
        assert status is None

    def test_empty_cache(self):
        """Test with empty cache."""
        cache = {}
        exists, status = check_run_status_from_cache("task_123", cache)
        assert exists is False
        assert status is None


class TestFetchWandbRunStatusCache:
    """Test fetch_wandb_run_status_cache function."""

    @patch("run_experiment.wandb.Api")
    def test_successful_fetch(self, mock_api_class):
        """Test successful fetching of runs from WandB."""
        # Setup mock
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_run1 = Mock()
        mock_run1.config.get.return_value = "task_1"
        mock_run1.state = "finished"
        mock_run1.url = "http://wandb.ai/run1"

        mock_run2 = Mock()
        mock_run2.config.get.return_value = "task_2"
        mock_run2.state = "running"
        mock_run2.url = "http://wandb.ai/run2"

        mock_api.runs.return_value = [mock_run1, mock_run2]

        # Execute
        cache = fetch_wandb_run_status_cache("exp_001", "entity", "project")

        # Verify
        assert len(cache) == 2
        assert cache["task_1"] == (True, "finished")
        assert cache["task_2"] == (True, "running")
        mock_api.runs.assert_called_once_with(
            "entity/project", filters={"config.exp_id": "exp_001"}
        )

    @patch("run_experiment.wandb.Api")
    def test_fetch_with_missing_entity(self, mock_api_class):
        """Test when entity or project is missing."""
        cache = fetch_wandb_run_status_cache("exp_001", None, "project")
        assert cache == {}

        cache = fetch_wandb_run_status_cache("exp_001", "entity", None)
        assert cache == {}

    @patch("run_experiment.wandb.Api")
    def test_fetch_with_no_runs(self, mock_api_class):
        """Test when no runs are found."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api
        mock_api.runs.return_value = []

        cache = fetch_wandb_run_status_cache("exp_001", "entity", "project")
        assert cache == {}

    @patch("run_experiment.wandb.Api")
    def test_fetch_with_runs_missing_task_id(self, mock_api_class):
        """Test when runs don't have task_id in config."""
        mock_api = Mock()
        mock_api_class.return_value = mock_api

        mock_run = Mock()
        mock_run.config.get.return_value = None  # No task_id
        mock_run.state = "finished"

        mock_api.runs.return_value = [mock_run]

        cache = fetch_wandb_run_status_cache("exp_001", "entity", "project")
        assert cache == {}  # Should skip runs without task_id


class TestDetectGpuMemoryPerDevice:
    """Test detect_gpu_memory_per_device function."""

    @patch("run_experiment.torch.cuda")
    def test_no_cuda_available(self, mock_cuda):
        """Test when CUDA is not available."""
        mock_cuda.is_available.return_value = False
        result = detect_gpu_memory_per_device()
        assert result == {}

    @patch("run_experiment.torch.cuda")
    def test_single_gpu_type(self, mock_cuda):
        """Test with single GPU type."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 4

        mock_props = Mock()
        mock_props.total_memory = 40 * (1024**3)  # 40 GB
        mock_props.name = "A100-40GB"
        mock_cuda.get_device_properties.return_value = mock_props

        result = detect_gpu_memory_per_device()
        assert 40 in result
        assert len(result[40]) == 4
        assert result[40] == [0, 1, 2, 3]

    @patch("run_experiment.torch.cuda")
    def test_mixed_gpu_types(self, mock_cuda):
        """Test with mixed GPU types."""
        mock_cuda.is_available.return_value = True
        mock_cuda.device_count.return_value = 4

        def mock_get_props(i):
            mock_props = Mock()
            if i < 2:
                mock_props.total_memory = 40 * (1024**3)  # 40 GB
                mock_props.name = "A100-40GB"
            else:
                mock_props.total_memory = 80 * (1024**3)  # 80 GB
                mock_props.name = "A100-80GB"
            return mock_props

        mock_cuda.get_device_properties.side_effect = mock_get_props

        result = detect_gpu_memory_per_device()
        assert 40 in result
        assert 80 in result
        assert result[40] == [0, 1]
        assert result[80] == [2, 3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Sample tests for the research template.

These tests serve as examples and can be extended for your research code.
"""

import pytest


def test_basic_arithmetic():
    assert 1 + 1 == 2


def test_imports():
    """Verify that common ML libraries can be imported."""
    import torch
    import transformers
    from datasets import Dataset

    assert torch.__version__ is not None
    assert transformers.__version__ is not None


@pytest.fixture
def sample_dataset():
    """Fixture providing a minimal dataset for testing."""
    from datasets import Dataset

    data = {
        "text": ["Hello world", "Test data", "Sample text"],
        "label": [0, 1, 0],
    }
    return Dataset.from_dict(data)


def test_dataset_fixture(sample_dataset):
    """Test that the sample dataset fixture works."""
    assert len(sample_dataset) == 3
    assert "text" in sample_dataset.column_names
    assert "label" in sample_dataset.column_names

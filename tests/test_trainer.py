"""Tests for the trainer module."""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from packages.trainer import Trainer


class MockModel(PreTrainedModel):
    """Mock model for testing."""

    def __init__(self):
        super().__init__(self.config)
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        """Mock forward pass."""
        outputs = self.linear(torch.randn(1, 10))
        loss = torch.nn.functional.cross_entropy(
            outputs, torch.tensor([1], dtype=torch.long)
        )
        return type("Outputs", (), {"loss": loss})()


class MockTokenizer(PreTrainedTokenizer):
    """Mock tokenizer for testing."""

    def __init__(self):
        super().__init__()
        self.vocab = {"[PAD]": 0, "hello": 1, "world": 2}

    def _tokenize(self, text, **kwargs):
        """Mock tokenization."""
        return text.split()

    def _convert_token_to_id(self, token):
        """Mock token to id conversion."""
        return self.vocab.get(token, 0)

    def _convert_id_to_token(self, index):
        """Mock id to token conversion."""
        for token, idx in self.vocab.items():
            if idx == index:
                return token
        return "[PAD]"

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """Mock special tokens addition."""
        return token_ids_0


@pytest.fixture
def trainer():
    """Create a trainer instance for testing."""
    model = MockModel()
    tokenizer = MockTokenizer()
    args = type(
        "Args",
        (),
        {
            "loss_type": "cross_entropy",
            "output_dir": "output",
            "num_train_epochs": 1,
            "per_device_train_batch_size": 2,
        },
    )
    return Trainer(model=model, args=args, tokenizer=tokenizer)


def test_trainer_initialization(trainer):
    """Test that trainer can be initialized."""
    assert trainer is not None
    assert trainer.model is not None
    assert trainer.args is not None
    assert trainer.loss_function is not None


def test_trainer_compute_loss(trainer):
    """Test the compute_loss method."""
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[1]], dtype=torch.long),
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_trainer_compute_loss_with_outputs(trainer):
    """Test the compute_loss method with return_outputs=True."""
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "labels": torch.tensor([[1]], dtype=torch.long),
    }
    loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    assert isinstance(loss, torch.Tensor)
    assert hasattr(outputs, "loss")
    assert torch.equal(loss, outputs.loss)

"""Tests for the trainer module."""

import dataclasses
from typing import Dict, List, Optional

import pytest
import torch
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from src.packages.trainer import Trainer


@dataclasses.dataclass
class CustomTrainingArguments(TrainingArguments):
    """Custom training arguments with additional fields."""

    loss_type: str = "cross_entropy"

    def __post_init__(self):
        super().__post_init__()
        self.no_cuda = True  # Force CPU for testing


class MockConfig(PretrainedConfig):
    """Mock config for testing."""

    model_type = "mock"


class MockModel(PreTrainedModel):
    """Mock model for testing."""

    def __init__(self):
        config = MockConfig()
        super().__init__(config)
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        """Mock forward pass."""
        batch_size = input_ids.shape[0] if input_ids is not None else 1
        device = input_ids.device if input_ids is not None else self.device
        outputs = self.linear(torch.randn(batch_size, 10, device=device))
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(outputs, labels.view(-1))
            return type("Outputs", (), {"loss": loss})()
        return type("Outputs", (), {"loss": None})()


class MockTokenizer(PreTrainedTokenizer):
    """Mock tokenizer for testing."""

    vocab_files_names = {"vocab_file": "vocab.txt"}

    def __init__(self):
        self._vocab = {"[PAD]": 0, "hello": 1, "world": 2}
        super().__init__(
            bos_token="[PAD]",
            eos_token="[PAD]",
            unk_token="[PAD]",
            pad_token="[PAD]",
            mask_token="[PAD]",
        )

    @property
    def vocab_size(self) -> int:
        """Size of vocabulary."""
        return len(self._vocab)

    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary mapping."""
        return self._vocab.copy()

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        """Mock tokenization."""
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        """Mock token to id conversion."""
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        """Mock id to token conversion."""
        for token, idx in self._vocab.items():
            if idx == index:
                return token
        return "[PAD]"

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """Mock special tokens addition."""
        return token_ids_0

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> List[str]:
        """Mock save vocabulary."""
        return []


@pytest.fixture
def trainer():
    """Create a trainer instance for testing."""
    model = MockModel()
    tokenizer = MockTokenizer()
    args = CustomTrainingArguments(
        output_dir="output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        loss_type="cross_entropy",
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
    device = trainer.model.device
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long, device=device),
        "labels": torch.tensor([[1]], dtype=torch.long, device=device),
    }
    loss = trainer.compute_loss(trainer.model, inputs)
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_trainer_compute_loss_with_outputs(trainer):
    """Test the compute_loss method with return_outputs=True."""
    device = trainer.model.device
    inputs = {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long, device=device),
        "labels": torch.tensor([[1]], dtype=torch.long, device=device),
    }
    loss, outputs = trainer.compute_loss(trainer.model, inputs, return_outputs=True)
    assert isinstance(loss, torch.Tensor)
    assert hasattr(outputs, "loss")
    assert torch.equal(loss, outputs.loss)

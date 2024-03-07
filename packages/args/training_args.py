from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HfTrainingArguments

from .base_args import BaseArguments


@dataclass
class TrainingArguments(HfTrainingArguments, BaseArguments):
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum position embedding per segment."},
    )

    def validate(self):
        pass

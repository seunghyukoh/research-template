from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments as HfTrainingArguments


@dataclass
class TrainingArguments(HfTrainingArguments):
    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum position embedding per segment."},
    )

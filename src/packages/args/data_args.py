from dataclasses import dataclass, field
from typing import Optional
from .base_args import BaseArguments


@dataclass
class DataArguments(BaseArguments):
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use"},
    )

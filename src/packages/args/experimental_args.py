from dataclasses import dataclass, field

from .base_args import BaseArguments


@dataclass
class ExperimentalArguments(BaseArguments):
    fast_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use fast attention or not (experimental)"},
    )

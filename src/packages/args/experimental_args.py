from dataclasses import dataclass, field


@dataclass
class ExperimentalArguments:
    fast_attention: bool = field(
        default=False,
        metadata={"help": "Whether to use fast attention or not (experimental)"},
    )

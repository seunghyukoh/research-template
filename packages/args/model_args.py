from dataclasses import dataclass, field
from typing import Optional

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING

from .base_args import BaseArguments

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments(BaseArguments):
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Model checkpoint for initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )

    def validate(self):
        assert (
            self.model_type in MODEL_TYPES
        ), "Model type should be one of: " + ", ".join(MODEL_TYPES)


if __name__ == "__main__":
    model_args = ModelArguments(
        model_name_or_path="gpt2",
        model_type="bart",
    )
    model_args.validate()

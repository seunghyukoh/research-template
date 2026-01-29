"""Training utilities for supervised fine-tuning."""

from typing import Optional

from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import SFTTrainer as BaseSFTTrainer


class SFTTrainer:
    """Wrapper for TRL SFTTrainer with common configurations."""

    def __init__(
        self,
        model_name: str,
        dataset,
        output_dir: str = "./outputs",
        use_lora: bool = True,
        lora_config: Optional[LoraConfig] = None,
    ):
        """Initialize SFT trainer.

        Args:
            model_name: HuggingFace model identifier
            dataset: Training dataset
            output_dir: Directory for saving checkpoints
            use_lora: Whether to use LoRA for parameter-efficient fine-tuning
            lora_config: Custom LoRA configuration (if None, uses defaults)
        """
        self.model_name = model_name
        self.dataset = dataset
        self.output_dir = output_dir
        self.use_lora = use_lora

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        if use_lora:
            if lora_config is None:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
            self.model = get_peft_model(self.model, lora_config)

        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            logging_steps=10,
            save_strategy="epoch",
            report_to="wandb",
        )

    def train(self, training_args: Optional[TrainingArguments] = None):
        """Start training.

        Args:
            training_args: Custom training arguments (overrides defaults)
        """
        if training_args is not None:
            self.training_args = training_args

        trainer = BaseSFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_args,
            train_dataset=self.dataset,
        )

        trainer.train()
        return trainer

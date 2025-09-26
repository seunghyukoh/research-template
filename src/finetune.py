import logging
import os
import sys
from pprint import pprint
from typing import Tuple

from accelerate import Accelerator
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
)

import wandb
from packages.args.finetune import DataArguments, FinetuneArguments, ModelArguments
from packages.datasets import DATASETS
from packages.models import load_model_and_tokenizer
from packages.utils.parse_args import parse_args

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_logging(log_file: str):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_file),  # Save log to file
            logging.StreamHandler(sys.stdout),  # Print log to stdout
        ],
    )

    logger = logging.getLogger()
    return logger


def get_model_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_args.model_path,
        tokenizer_name_or_path=model_args.tokenizer_path,
    )

    # Update tokenizer
    if model_args.chat_template is not None:
        tokenizer.chat_template = model_args.chat_template

    return model, tokenizer


def set_peft(model_args: ModelArguments, model: PreTrainedModel):
    if isinstance(model, PeftModel) or not model_args.use_lora:
        return model

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=model_args.lora_rank,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
    )

    model = get_peft_model(model, lora_config)
    logger.info("Applied LoRA to model.")

    model.print_trainable_parameters()

    # for checkpointing
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    return model


def get_dataset(
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizer,
):
    dataset_cls = DATASETS[data_args.dataset_name]
    datasets = dataset_cls(
        tokenizer,
        dataset_path=data_args.dataset_path,
        max_train_samples=data_args.max_train_samples,
        max_validation_samples=data_args.max_validation_samples,
        max_test_samples=data_args.max_validation_samples,
        shuffle_seed=data_args.shuffle_seed,
    )

    return datasets.train, datasets.validation


def main(args: FinetuneArguments):
    data_args = args.data_args
    model_args = args.model_args
    training_args = args.training_args
    # Set seed
    set_seed(training_args.seed)

    # Load tokenizer and model
    with Accelerator().main_process_first():
        model, tokenizer = get_model_and_tokenizer(model_args)

    # Set Peft
    model = set_peft(model_args, model)

    # Log the number of trainable model params
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"trainable model_params: {model_params}")

    # Log model
    if Accelerator().is_main_process:
        pprint(model)

    # Load dataset
    with Accelerator().main_process_first():
        train_dataset, validation_dataset = get_dataset(data_args, tokenizer)

    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    )

    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    load_dotenv()
    # Parse arguments
    args: DataArguments = parse_args(FinetuneArguments)

    logger = setup_logging(args.training_args.log_file)

    accelerator = Accelerator()

    is_main_process = accelerator.is_main_process
    if is_main_process:
        raw_args = args.to_dict()

        # Print arguments
        pprint(raw_args)

        wandb.init(
            id=args.experiment_args.id,
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            name=args.training_args.run_name,
            config=raw_args,
            tags=args.experiment_args.wandb_tags,
            group=args.experiment_args.wandb_group,
        )

    accelerator.wait_for_everyone()

    main(args)

import logging
import os
from pprint import pprint
from typing import Tuple

import torch
import wandb
from accelerate import Accelerator
from dotenv import load_dotenv
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from tokenizers.processors import TemplateProcessing
from transformers import (
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    set_seed,
)

from packages.args.finetune import (
    DataArguments,
    FinetuneArguments,
    ModelArguments,
)
from packages.datasets import DATASETS
from packages.models import load_model_and_tokenizer
from packages.utils.parse_args import parse_args


def get_model_and_tokenizer(
    model_args: ModelArguments,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()  # Apple Silicon
        else "cpu"
    )

    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=model_args.model_path,
        tokenizer_name_or_path=model_args.tokenizer_path,
        model_kwargs=dict(
            device_map=device,
            use_cache=False,
        ),
    )

    # Update tokenizer
    if model_args.chat_template is not None:
        tokenizer.chat_template = model_args.chat_template

    if model_args.pad_token is not None:
        tokenizer.add_special_tokens(dict(pad_token=model_args.pad_token))
    if model_args.eos_token is not None:
        tokenizer.add_special_tokens(dict(eos_token=model_args.eos_token))
    if model_args.bos_token is not None:
        tokenizer.add_special_tokens(dict(bos_token=model_args.bos_token))

    if model_args.tokenizer_post_processor_single is not None:
        tokenizer._tokenizer.post_processor = TemplateProcessing(
            single=model_args.tokenizer_post_processor_single,
            special_tokens=[
                (tokenizer.bos_token, tokenizer.bos_token_id),
                (tokenizer.eos_token, tokenizer.eos_token_id),
            ],
        )

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
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, padding="longest"
        ),
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

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Parse arguments
    args: DataArguments = parse_args(FinetuneArguments)

    accelerator = Accelerator()

    is_main_process = accelerator.is_main_process
    if is_main_process:
        raw_args = args.to_dict()

        # Print arguments
        pprint(raw_args)

        wandb.init(
            id=args.uuid,
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            name=args.training_args.run_name,
            config=raw_args,
            tags=args.experiment_args.wandb_tags,
            group=args.experiment_args.wandb_group,
        )

    accelerator.wait_for_everyone()

    main(args)

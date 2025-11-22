from copy import deepcopy

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import get_logger
from trl import ModelConfig, SFTConfig, SFTTrainer

import wandb
from utils.batch_size import AUTO_BATCH_SIZE_TRAIN_STEPS, get_max_batch_size
from utils.dataset_preprocessing import get_preprocessing_fn
from utils.hydra_decorators import hydra_main_with_logging
from utils.parse_args import Parser

logger = get_logger(__name__)

load_dotenv()


def load_model(model_args: ModelConfig):
    dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load model
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
        device_map=device,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
    )
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def load_and_preprocess_datasets(dataset_args: DictConfig):
    train_dataset = load_dataset(
        dataset_args.dataset_name,
        name=dataset_args.dataset_config,
        split=dataset_args.dataset_train_split,
    )
    eval_dataset = load_dataset(
        dataset_args.dataset_name,
        name=dataset_args.dataset_config,
        split=dataset_args.dataset_eval_split,
    )
    assert isinstance(train_dataset, Dataset), f"Train dataset is not a Dataset: {train_dataset}"
    assert isinstance(eval_dataset, Dataset), f"Eval dataset is not a Dataset: {eval_dataset}"

    dataset = DatasetDict(
        {
            "train": train_dataset,
            "eval": eval_dataset,
        }
    )

    # Get preprocessing function from registry
    preprocessing_fn_name = getattr(dataset_args, "preprocessing_fn", None)
    if preprocessing_fn_name is None:
        raise ValueError(
            "Dataset configuration must specify 'preprocessing_fn'. "
            "Please add it to your dataset config file."
        )

    try:
        preprocessing_fn = get_preprocessing_fn(preprocessing_fn_name)
    except ValueError as e:
        raise ValueError(
            f"Failed to load preprocessing function '{preprocessing_fn_name}': {e}"
        ) from e

    dataset = dataset.map(preprocessing_fn)

    return dataset


# Loads config from `configs/sft/config.yaml`
@hydra_main_with_logging(config_path="configs/sft", config_name="config")
def main(cfg: DictConfig):
    accelerator = Accelerator()

    # Update this to parse customized arguments
    parser = Parser([SFTConfig, ModelConfig])
    [training_args, model_args] = parser.parse_dict(
        {
            **OmegaConf.to_container(cfg.training, resolve=True),
            **OmegaConf.to_container(cfg.model, resolve=True),
        }
    )

    accelerator.wait_for_everyone()

    model, tokenizer = load_model(model_args)

    # Load dataset
    dataset = load_and_preprocess_datasets(cfg.dataset)

    def demo_run_with_batch_size(batch_size=128):
        model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        original_training_mode = model.training

        training_args_copy = deepcopy(training_args)
        training_args_copy.report_to = "none"
        training_args_copy.per_device_train_batch_size = batch_size
        training_args_copy.max_steps = AUTO_BATCH_SIZE_TRAIN_STEPS
        training_args_copy.save_strategy = "no"
        training_args_copy.push_to_hub = False

        max_length = cfg.training.max_length  # 512
        demo_dataset = Dataset.from_dict(
            # Create dummy data for demonstration
            {
                "input_ids": [
                    torch.zeros(max_length, dtype=torch.int64) for _ in range(batch_size)
                ],
                "attention_mask": [
                    torch.ones(max_length, dtype=torch.int64) for _ in range(batch_size)
                ],
            }
        )  # [batch_size, max_length]

        # Train
        demo_trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            args=training_args_copy,
            train_dataset=demo_dataset,
            eval_dataset=demo_dataset,
        )

        try:
            demo_trainer.train()
        finally:
            del demo_trainer
            del demo_dataset

            model.load_state_dict(model_state_dict)
            model.train(original_training_mode)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def get_proper_batch_size():

        effective_per_device_batch_size = (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        )

        batch_size = get_max_batch_size(
            lambda bs: demo_run_with_batch_size(bs),
            starting_batch_size=training_args.per_device_train_batch_size,
        )
        gradient_accumulation_steps = max(effective_per_device_batch_size // batch_size, 1)

        return batch_size, gradient_accumulation_steps

    try:
        batch_size, gradient_accumulation_steps = get_proper_batch_size()
        logger.info(
            f"Batch size: {batch_size}, Gradient accumulation steps: {gradient_accumulation_steps}"
        )
        training_args.per_device_train_batch_size = batch_size
        training_args.per_device_eval_batch_size = batch_size
        training_args.gradient_accumulation_steps = gradient_accumulation_steps
    except RuntimeError as e:
        logger.warning(
            f"Failed to find optimal batch size automatically: {e}. "
            f"Using default values: batch_size={training_args.per_device_train_batch_size}, "
            f"gradient_accumulation_steps={training_args.gradient_accumulation_steps}"
        )

    if wandb.run is not None and accelerator.is_main_process:
        # Update wandb config with the parsed model arguments
        wandb.config.update(
            {
                **training_args.__dict__,
                **model_args.__dict__,
            },
            allow_val_change=True,
        )

    # Train
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )

    if cfg.debug.dry_run:
        logger.info("Dry run: Skipping training")
        logger.info("Training arguments:")
        logger.info(training_args)
        logger.info("Model arguments:")
        logger.info(model_args)
        logger.info("Dataset:")
        logger.info(dataset)
    else:
        trainer.train()


if __name__ == "__main__":
    main()

from copy import deepcopy

import torch
from accelerate import Accelerator
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import get_logger
from trl import ModelConfig, SFTConfig, SFTTrainer

from utils.batch_size import AUTO_BATCH_SIZE_TRAIN_STEPS, optimize_batch_size
from utils.dataset_preprocessing import get_preprocessing_fn
from utils.hydra_decorators import hydra_main_with_logging
from utils.parse_args import Parser

logger = get_logger(__name__)

load_dotenv()


def get_device() -> str:
    """Return available device"""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def load_model(model_args: ModelConfig):
    dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )
    device = get_device()

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


def handle_dry_run(
    cfg: DictConfig, training_args: SFTConfig, model_args: ModelConfig, dataset: DatasetDict
):
    """Logging for dry run"""
    logger.info("Dry run: Skipping training")
    logger.info("Training arguments:")
    logger.info(training_args)
    logger.info("Model arguments:")
    logger.info(model_args)
    logger.info("Dataset:")
    logger.info(dataset)


def create_demo_run_fn(
    model,
    tokenizer,
    training_args: SFTConfig,
    max_length: int,
):
    """Create demo run function for batch size test

    This function can be implemented differently for each experiment.
    For example, different trainer or different dataset can be used.
    Args:
        model: Model
        tokenizer: Tokenizer
        training_args: Training arguments
        max_length: Maximum length of the input
    Returns:
        Callable[[int], None]: A function that takes a batch size (int) as input,
        runs a demo training step with that batch size, and returns None.
    Raises:
        RuntimeError: If the batch size is not found
    Example:
        >>> demo_run_fn = create_demo_run_fn(model, tokenizer, training_args, max_length)
        >>> demo_run_fn(batch_size)
    """

    def demo_run_with_batch_size(batch_size: int):
        model_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        original_training_mode = model.training

        training_args_copy = deepcopy(training_args)
        training_args_copy.report_to = "none"
        training_args_copy.per_device_train_batch_size = batch_size
        training_args_copy.max_steps = AUTO_BATCH_SIZE_TRAIN_STEPS
        training_args_copy.save_strategy = "no"
        training_args_copy.push_to_hub = False

        demo_dataset = Dataset.from_dict(
            {
                "input_ids": [
                    torch.zeros(max_length, dtype=torch.int64) for _ in range(batch_size)
                ],
                "attention_mask": [
                    torch.ones(max_length, dtype=torch.int64) for _ in range(batch_size)
                ],
            }
        )

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

    return demo_run_with_batch_size


def optimize_training_batch_size(
    model,
    tokenizer,
    training_args: SFTConfig,
    max_length: int,
) -> None:
    """Find optimal batch size and update training_args
    Args:
        model: Model
        tokenizer: Tokenizer
        training_args: Training arguments
        max_length: Maximum length of the input
    Returns:
        None
    Raises:
        RuntimeError: If the batch size is not found or the gradient accumulation steps are not found
    """

    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )

    demo_run_fn = create_demo_run_fn(model, tokenizer, training_args, max_length)

    try:
        batch_size, gradient_accumulation_steps = optimize_batch_size(
            demo_run_fn=demo_run_fn,
            effective_batch_size=effective_batch_size,
            starting_batch_size=training_args.per_device_train_batch_size,
        )
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


# Loads config from `configs/sft/config.yaml`
@hydra_main_with_logging(config_path="configs/sft", config_name="config")
def main(cfg: DictConfig):
    accelerator = Accelerator()

    # Parse arguments
    parser = Parser([SFTConfig, ModelConfig])
    [training_args, model_args] = parser.parse_dict(
        {
            **OmegaConf.to_container(cfg.training, resolve=True),
            **OmegaConf.to_container(cfg.model, resolve=True),
        }
    )

    accelerator.wait_for_everyone()

    # Load model and tokenizer
    model, tokenizer = load_model(model_args)

    # Load and preprocess datasets
    dataset = load_and_preprocess_datasets(cfg.dataset)

    # Optimize batch size
    optimize_training_batch_size(model, tokenizer, training_args, cfg.training.max_length)

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )

    # Train or dry run
    if cfg.debug.dry_run:
        handle_dry_run(cfg, training_args, model_args, dataset)
    else:
        trainer.train()


if __name__ == "__main__":
    main()

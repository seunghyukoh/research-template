"""Main script for fine-tuning models."""

import logging
import os
import sys
from pprint import pprint
from typing import Tuple

import wandb
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

from packages.args.finetune import (
    DataArguments,
    FinetuneArguments,
    ModelArguments,
)
from packages.datasets import DATASETS
from packages.models import load_model_and_tokenizer
from packages.utils.exceptions import (
    ConfigurationError,
    DatasetLoadError,
    ModelLoadError,
    ResourceError,
)
from packages.utils.logging import PerformanceMonitor, log_system_info, setup_logging
from packages.utils.parse_args import parse_args
from packages.utils.validation import (
    validate_dataset_config,
    validate_model_config,
    validate_system_requirements,
    validate_training_config,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Setup basic logging for errors before main logger is configured
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.ERROR,
    handlers=[logging.StreamHandler(sys.stderr)],
)
error_logger = logging.getLogger("error_handler")


def get_model_and_tokenizer(
    model_args: ModelArguments,
    logger: logging.Logger,
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer.

    Args:
        model_args: Model arguments
        logger: Logger instance

    Returns:
        Tuple of (model, tokenizer)

    Raises:
        ModelLoadError: If model or tokenizer loading fails
    """
    try:
        logger.info(f"Loading model from {model_args.model_path}")
        model, tokenizer = load_model_and_tokenizer(
            model_name_or_path=model_args.model_path,
            tokenizer_name_or_path=model_args.tokenizer_path,
        )

        # Update tokenizer
        if model_args.chat_template is not None:
            try:
                tokenizer.chat_template = model_args.chat_template
                logger.info("Applied chat template to tokenizer")
            except Exception as e:
                raise ModelLoadError(f"Failed to set chat template: {str(e)}") from e

        return model, tokenizer
    except Exception as e:
        raise ModelLoadError(f"Failed to load model or tokenizer: {str(e)}") from e


def set_peft(
    model_args: ModelArguments,
    model: PreTrainedModel,
    logger: logging.Logger,
) -> PreTrainedModel:
    """Apply PEFT to model.

    Args:
        model_args: Model arguments
        model: Model to apply PEFT to
        logger: Logger instance

    Returns:
        Model with PEFT applied

    Raises:
        ConfigurationError: If PEFT configuration is invalid
    """
    if isinstance(model, PeftModel) or not model_args.use_lora:
        return model

    try:
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
    except Exception as e:
        raise ConfigurationError(f"Failed to apply PEFT: {str(e)}") from e


def get_dataset(
    data_args: DataArguments,
    tokenizer: PreTrainedTokenizer,
    logger: logging.Logger,
):
    """Load dataset.

    Args:
        data_args: Dataset arguments
        tokenizer: Tokenizer instance
        logger: Logger instance

    Returns:
        Tuple of (train_dataset, validation_dataset)

    Raises:
        DatasetLoadError: If dataset loading fails
    """
    try:
        dataset_cls = DATASETS[data_args.dataset_name]
        datasets = dataset_cls(
            tokenizer,
            dataset_path=data_args.dataset_path,
            max_train_samples=data_args.max_train_samples,
            max_validation_samples=data_args.max_validation_samples,
            max_test_samples=data_args.max_validation_samples,
            shuffle_seed=data_args.shuffle_seed,
        )

        logger.info(
            f"Dataset loaded: "
            f"Train={len(datasets.train)} samples, "
            f"Validation={len(datasets.validation)} samples"
        )

        return datasets.train, datasets.validation
    except Exception as e:
        raise DatasetLoadError(f"Failed to load dataset: {str(e)}") from e


def setup_wandb(
    args: FinetuneArguments, is_main_process: bool, logger: logging.Logger
) -> None:
    """Setup Weights & Biases logging.

    Args:
        args: Training arguments
        is_main_process: Whether this is the main process
        logger: Logger instance
    """
    if is_main_process:
        raw_args = args.to_dict()
        logger.info("Initializing Weights & Biases")
        wandb.init(
            id=args.experiment_args.id,
            project=os.getenv("WANDB_PROJECT", "huggingface"),
            name=args.training_args.run_name,
            config=raw_args,
            tags=args.experiment_args.wandb_tags,
            group=args.experiment_args.wandb_group,
        )


def main(args: FinetuneArguments, logger: logging.Logger):
    """Main training function.

    Args:
        args: Training arguments
        logger: Logger instance
    """
    data_args = args.data_args
    model_args = args.model_args
    training_args = args.training_args

    # Validate configurations
    validate_model_config(model_args.to_dict())
    validate_training_config(training_args.to_dict())
    validate_dataset_config(data_args.to_dict(), allowed_datasets=list(DATASETS.keys()))

    # Check system requirements (adjust values as needed)
    # validate_system_requirements(
    #     required_memory_gb=16.0,
    #     required_gpu_memory_gb=8.0,
    #     required_gpu_count=1,
    # )

    # Set seed
    set_seed(training_args.seed)
    logger.info(f"Set random seed to {training_args.seed}")

    # Load tokenizer and model
    with Accelerator().main_process_first():
        model, tokenizer = get_model_and_tokenizer(model_args, logger)

    # Set Peft
    model = set_peft(model_args, model, logger)

    # Log the number of trainable model params
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable model parameters: {model_params:,}")

    # Log model architecture
    if Accelerator().is_main_process:
        logger.info("Model architecture:")
        logger.info(model)

    # Load dataset
    with Accelerator().main_process_first():
        train_dataset, validation_dataset = get_dataset(data_args, tokenizer, logger)

    # Initialize performance monitor
    performance_monitor = PerformanceMonitor(
        logger, log_interval=training_args.logging_steps
    )

    # Create trainer
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

    # Add performance monitoring to trainer
    original_training_step = trainer.training_step

    def training_step_with_monitoring(*args, **kwargs):
        loss = original_training_step(*args, **kwargs)
        performance_monitor.step_complete(
            loss.item(),
            training_args.per_device_train_batch_size,
        )
        return loss

    trainer.training_step = training_step_with_monitoring

    try:
        # Training
        logger.info("Starting training")
        train_result = trainer.train()
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("Training completed successfully!")
        logger.info("Final metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    load_dotenv()

    try:
        # Parse arguments
        args: FinetuneArguments = parse_args(FinetuneArguments)

        # Setup logging
        logger = setup_logging(
            args.training_args.log_file,
            log_level=args.training_args.log_level
            or "INFO",  # Default to INFO if not specified
        )

        # Log system information
        log_system_info(logger)

        accelerator = Accelerator()
        setup_wandb(args, accelerator.is_main_process, logger)

        # Print arguments
        if accelerator.is_main_process:
            logger.info("Training arguments:")
            pprint(args.to_dict())

        accelerator.wait_for_everyone()

        main(args, logger)

    except Exception as e:
        # Use error_logger if main logger setup failed
        log = error_logger
        log.error(f"Fatal error: {str(e)}", exc_info=True)
        sys.exit(1)

import logging
import os
import sys
from typing import Any, Dict, Tuple

import datasets
import transformers
from torch import TensorType
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    OPTForCausalLM,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    Trainer,
    set_seed,
)

from args import DataArguments, ExperimentalArguments, ModelArguments, TrainingArguments
from packages.data_utils.datasets import load_preprocessed_datasets, load_raw_datasets
from packages.data_utils.preprocess import preprocess_datasets

logger = logging.getLogger(__name__)

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast
Dataset = Dict["str", TensorType]


def parse_args() -> (
    Tuple[
        ModelArguments,
        DataArguments,
        TrainingArguments,
        ExperimentalArguments,
    ]
):
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ExperimentalArguments)
    )

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        (
            model_args,
            data_args,
            training_args,
            experimental_args,
        ) = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
            experimental_args,
        ) = parser.parse_args_into_dataclasses()

    return model_args, data_args, training_args, experimental_args


def set_logger(log_level: Any) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set logging level
    logger.setLevel(log_level)

    # Set logging level for datasets and transformers
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


def prepare_tokenizer(model_args: ModelArguments) -> Tokenizer:
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.tokenizer_name:
        tokeninzer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, **tokenizer_kwargs
        )
    elif model_args.model_name_or_path:
        tokeninzer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported in this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    return tokeninzer


def prepare_config(model_args: ModelArguments) -> Any:
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }

    if model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    elif model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        config.update(**config_kwargs)
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.config_overrides is not None:
        logger.info(f"Overriding config: {model_args.config_overrides}")
        config.update_from_string(model_args.config_overrides)
        logger.info(f"New config: {config}")

    return config


def prepare_model(model_args: ModelArguments, config: Any) -> OPTForCausalLM:
    if model_args.model_name_or_path:
        model = OPTForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
        )
    else:
        model = OPTForCausalLM.from_config(config)
        n_params = sum(
            dict((p.data_ptr(), p.numel()) for p in model.parameters()).values()
        )
        logger.info(
            f"Training new model from scratch - Total size={n_params/2**20:.2f}M parameters"
        )

    return model


def prepare_datasets(
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: TrainingArguments,
    tokenizer: Tokenizer,
) -> Tuple[Dataset, Dataset]:
    do_train = training_args.do_train
    if not do_train:
        data_args.preprocess_train_datasets = []

    use_preprocessed_datasets = len(data_args.preprocessed_validation_datasets) > 0
    if use_preprocessed_datasets:
        lm_datasets = load_preprocessed_datasets(data_args, model_args)
    else:
        raw_datasets = load_raw_datasets(data_args, model_args)
        lm_datasets = preprocess_datasets(
            raw_datasets, tokenizer, data_args, training_args
        )

    if do_train:
        if "train" not in lm_datasets:
            raise ValueError("--do_train requires a train dataset.")
        train_dataset = lm_datasets["train"]

        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        print(f"Total number of training samples: {len(train_dataset)}")

    do_eval = training_args.do_eval
    if do_eval:
        eval_dataset = {}
        for key in lm_datasets.keys():
            if "validation" in key:
                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(
                        data_args.max_eval_samples, len(lm_datasets[key])
                    )
                    eval_dataset[key] = lm_datasets[key].select(range(max_eval_samples))
                else:
                    eval_dataset[key] = lm_datasets[key]

    return train_dataset, eval_dataset


def main():
    model_args, data_args, training_args, experimental_args = parse_args()

    # Set logger
    log_level = training_args.get_process_log_level()
    set_logger(log_level)

    # Log arguments
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu} "
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detect last checkpoint
    last_checkpoint = None
    if training_args.resume_from_checkpoint:
        last_checkpoint = get_last_checkpoint_or_last_model(training_args.output_dir)
        last_checkpoint = None

        if last_checkpoint is None:
            print(
                f"Didn't find any checkpoint to resume training from in {training_args.output_dir}. Starting training from scratch."
            )

        else:
            print(
                f"Found checkpoint {last_checkpoint}. Using this checkpoint for resuming training."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Prepare tokenizer
    tokenizer = prepare_tokenizer(model_args)

    # Prepare config
    config = prepare_config(model_args)
    # Update config with experimental arguments if provided
    # Here ->

    # Prepare model
    model = prepare_model(model_args, config)

    # Prepare datasets
    train_dataset, eval_dataset = prepare_datasets(
        data_args, model_args, training_args, tokenizer
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    if last_checkpoint is not None:
        trainer._load_from_checkpoint(last_checkpoint)
    else:
        logger.info("Training new model from scratch")

    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        if training_args.do_train:
            metrics["global_step"] = trainer.state.global_step
        else:
            if last_checkpoint is None:
                metrics["global_step"] = 0
                last_checkpoint = training_args.output_dir
            else:
                metrics["global_step"] = parse_checkpoint_step(last_checkpoint)
        metrics["model_name"] = last_checkpoint

        if training_args.do_train:
            trainer.log_metrics("eval")
            trainer.save_metrics("eval")

        else:
            if last_checkpoint is not None:
                step = parse_checkpoint_step(last_checkpoint)
            else:
                step = 0
            trainer.log_metrics(f"eval_step{step}")
            trainer.save_metrics(f"eval_step{step}")


if __name__ == "__main__":
    main()

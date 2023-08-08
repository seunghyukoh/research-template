import logging
import os
import sys
from typing import Tuple

import datasets
import transformers
from transformers import AutoTokenizer, HfArgumentParser, set_seed

from args import DataArguments, ExperimentalArguments, ModelArguments, TrainingArguments

logger = logging.getLogger(__name__)


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


def set_logger(log_level):
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


def prepare_tokenizer(model_args):
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
        # TODO
        # last_checkpoint = get_last_checkpoint_or_last_model(training_args.output_dir)
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

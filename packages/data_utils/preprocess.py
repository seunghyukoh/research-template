import logging
import math
from itertools import chain

import datasets
import transformers
from transformers.testing_utils import CaptureLogger

logger = logging.getLogger(__name__)


def _tokenize(
    tokenizer,
    examples,
    text_column_name,
    block_size,
    add_special_tokens,
    tok_logger,
    pad=False,
):
    texts = []

    for text in examples[text_column_name]:
        while text.startswith("</s>"):
            text = text[len("</s>") :]
        texts.append(text)

    with CaptureLogger(tok_logger) as cl:
        if pad:
            output = tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=block_size,
                add_special_tokens=add_special_tokens,
            )
        else:
            output = tokenizer(
                texts,
                add_special_tokens=add_special_tokens,
            )
    if "Token indices sequence length is longer than the" in cl.out:
        tok_logger.warning(
            "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into small bits"
            " before being passed to the model."
        )

    return output


def _group_texts(examples, block_size):
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size

    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result


def preprocess_datasets(raw_datasets, tokenizer, data_args, training_args):
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    else:
        column_names = raw_datasets["validation"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    if not training_args.add_special_tokens:
        print("Removing special tokens in tokenization")

    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = datasets.DatasetDict()
        for key in raw_datasets.keys():
            pad = False
            if training_args.line_by_line_training and (
                key == "train" or key == "validation"
            ):
                pad = True

            tokenized_datasets[key] = raw_datasets[key].map(
                lambda x: _tokenize(
                    tokenizer=tokenizer,
                    examples=x,
                    text_column_name=text_column_name,
                    block_size=data_args.block_size,
                    add_special_tokens=training_args.add_special_tokens,
                    tok_logger=tok_logger,
                    pad=pad,
                ),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Tokenizing texts...",
            )

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({block_size}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
            block_size = 1024

    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    with training_args.main_process_first(desc="grouping texts together"):
        if training_args.line_by_line_training:
            for key in tokenized_datasets.keys():
                if "validation" in key and key != "validation":
                    tokenized_datasets[key] = tokenized_datasets[key].map(
                        lambda x: _group_texts(x, block_size),
                        batched=True,
                        num_proc=data_args.preprocessing_num_workers,
                        load_from_cache_file=not data_args.overwrite_cache,
                        desc="Grouping texts together",
                    )
                else:
                    tokenized_datasets[key] = tokenized_datasets[key].map(
                        lambda x: {"labels": x["input_ids"].copy()},
                        load_from_cache_file=not data_args.overwrite_cache,
                    )
            lm_datasets = tokenized_datasets
        else:
            lm_datasets = tokenized_datasets.map(
                lambda x: _group_texts(x, block_size),
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    if training_args.save_logits and training_args.train_data_index is not None:
        lens = len(lm_datasets["train"])
        num_data_per_index = math.ceil(training_args.train_data_percentage * lens)
        start_index = training_args.train_data_index * num_data_per_index
        end_index = min((training_args.train_data_index + 1) * num_data_per_index, lens)
        lm_datasets["train"] = lm_datasets["train"].select(
            range(start_index, end_index)
        )
        print(f"Total number of training data: {lens}")
        print(
            f"Training data index: {training_args.train_data_index}, start index: {start_index}, end index: {end_index}"
        )

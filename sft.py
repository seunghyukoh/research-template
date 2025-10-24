import os

import torch
from datasets import Dataset, load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, ScriptArguments, SFTConfig, SFTTrainer

import wandb
from utils.args import WandbConfig
from utils.parse_args import Parser

load_dotenv()


def load_model(model_config):
    dtype = (
        model_config.dtype
        if model_config.dtype in ["auto", None]
        else getattr(torch, model_config.dtype)
    )
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        dtype=dtype,
        device_map=device,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
        **model_kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        trust_remote_code=model_config.trust_remote_code,
    )
    return model, tokenizer


def load_and_preprocess_datasets(script_args):
    train_dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_train_split,
    )
    test_dataset = load_dataset(
        script_args.dataset_name,
        name=script_args.dataset_config,
        split=script_args.dataset_test_split,
    )
    assert isinstance(train_dataset, Dataset) and isinstance(test_dataset, Dataset)

    def routine(example):
        # Make this more generalizable
        prompt = example["question"]
        completion = example["answer"]
        return {"prompt": prompt, "completion": completion}

    train_dataset = train_dataset.map(routine)
    test_dataset = test_dataset.map(routine)

    return train_dataset, test_dataset


def dryrun(batch_size: int, sft_config: SFTConfig, model_config: ModelConfig):
    import numpy as np

    sft_config.max_steps = 1
    sft_config.push_to_hub = False
    sft_config.report_to = "none"
    sft_config.per_device_train_batch_size = batch_size
    sft_config.per_device_eval_batch_size = batch_size
    sft_config.gradient_accumulation_steps = 1
    sft_config.disable_tqdm = True

    model, tokenizer = load_model(model_config)

    max_length = sft_config.max_length
    # Create dummy data for demonstration
    data = {
        "input_ids": [np.zeros(max_length, dtype=np.int64) for _ in range(batch_size)],
        "attention_mask": [np.ones(max_length, dtype=np.int64) for _ in range(batch_size)],
    }
    train_dataset = Dataset.from_dict(data)

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
    )
    trainer.train()

    return batch_size


def main(sft_config, script_args, model_config, wandb_config):

    ################
    # Model & Tokenizer
    ################
    model, tokenizer = load_model(model_config)

    ################
    # Dataset
    ################
    train_dataset, test_dataset = load_and_preprocess_datasets(script_args)

    if os.environ.get("RANK", "0") == "0":
        wandb.init(
            id=wandb_config.id,
            entity=wandb_config.entity,
            project=wandb_config.project,
            name=sft_config.run_name,
            notes=wandb_config.notes,
            tags=wandb_config.tags,
            group=wandb_config.group,
            job_type=wandb_config.job_type,
            mode=wandb_config.mode,
            resume=wandb_config.resume,
            resume_from=wandb_config.resume_from,
            save_code=wandb_config.save_code,
            config=dict(
                **sft_config.__dict__,
                **script_args.__dict__,
                **model_config.__dict__,
            ),
        )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    trainer.train()

    # Save and push to hub
    trainer.save_model(sft_config.output_dir)
    if sft_config.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":

    parser = Parser(list([SFTConfig, ScriptArguments, ModelConfig, WandbConfig]))
    sft_config, script_args, model_config, wandb_config = parser.parse_args_into_dataclasses(
        args_file_flag="--config_file"
    )

    def get_proper_batch_size():
        from copy import deepcopy

        from utils.batch_size import get_max_batch_size

        effective_per_device_batch_size = (
            sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps
        )

        batch_size = get_max_batch_size(
            lambda bs: dryrun(
                bs, sft_config=deepcopy(sft_config), model_config=deepcopy(model_config)
            ),
            starting_batch_size=sft_config.per_device_train_batch_size,
        )
        gradient_accumulation_steps = max(effective_per_device_batch_size // batch_size, 1)

        return batch_size, gradient_accumulation_steps

    batch_size, gradient_accumulation_steps = get_proper_batch_size()
    sft_config.per_device_train_batch_size = batch_size
    sft_config.per_device_eval_batch_size = batch_size
    sft_config.gradient_accumulation_steps = gradient_accumulation_steps
    main(sft_config, script_args, model_config, wandb_config)

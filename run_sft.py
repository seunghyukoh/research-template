import torch
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ModelConfig, SFTConfig, SFTTrainer

import wandb
from utils.hydra_decorators import hydra_main_with_logging
from utils.parse_args import Parser

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


def load_and_preprocess_datasets(dataset_args):
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

    def routine(example):
        # Make this more generalizable
        if dataset_args.dataset_name == "openai/gsm8k":
            prompt = example["question"]
            completion = example["answer"]
            return {"prompt": prompt, "completion": completion}
        else:
            raise ValueError(f"Dataset {dataset_args.dataset_name} not supported")

    dataset = dataset.map(routine)

    return dataset


# Loads config from `configs/sft/config.yaml`
@hydra_main_with_logging(config_path="configs/sft", config_name="config")
def main(cfg: DictConfig):
    # Update this to parse customized arguments
    parser = Parser([SFTConfig, ModelConfig])
    [training_args, model_args] = parser.parse_dict({**cfg.training, **cfg.model})

    if wandb.run is not None:  # rank 0 only
        # Update wandb config with the parsed training and model arguments
        wandb.config.update(
            {
                "training": training_args.to_dict(),
                "model": model_args.to_dict(),
            },
            allow_val_change=True,
        )

    model, tokenizer = load_model(model_args)

    # Load dataset
    dataset = load_and_preprocess_datasets(cfg.dataset)

    # Train
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["eval"],
    )

    trainer.train()


if __name__ == "__main__":
    main()

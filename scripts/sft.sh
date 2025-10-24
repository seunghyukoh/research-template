#!/bin/bash

set -e

NUM_PROCESSES=1

accelerate launch \
--num_processes=$NUM_PROCESSES \
--num_machines=1 \
--mixed_precision="bf16" \
--dynamo_backend="inductor" \
sft.py \
--config_file configs/sft/script_args.yaml \
--config_file configs/sft/model_config.yaml \
--config_file configs/sft/sft_config.yaml \
--tags debug sft test \
--notes "sft test run" \
--output_dir outputs/sft \
--max_steps 100 \
--eval_steps 50 \
--per_device_train_batch_size 64 \
--per_device_eval_batch_size 64 \
--gradient_accumulation_steps 1 \
--push_to_hub true \
--hub_model_id JakeOh/debug \
--hub_strategy checkpoint \
--hub_private_repo true

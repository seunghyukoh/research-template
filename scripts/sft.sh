#!/bin/bash

set -e

accelerate launch src/sft.py \
--config_file experiments/sft/script_args.yaml \
--config_file experiments/sft/model_config.yaml \
--config_file experiments/sft/sft_config.yaml \
--tags debug sft test \
--notes "sft test run" \
--output_dir outputs/sft \
--max_steps 100 \
--eval_steps 50 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 16 \
--gradient_accumulation_steps 8 \
--push_to_hub true \
--hub_model_id JakeOh/debug \
--hub_strategy every_save \
--hub_private_repo true

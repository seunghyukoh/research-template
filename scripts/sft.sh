#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: ./sft.sh [num_processes] [max_steps]
# Example: ./sft.sh 4 1000

set -e

source .env

NUM_PROCESSES=${1:-1}
MAX_STEPS=${2:-100}

AUTO_BATCH_SIZE=true
BATCH_SIZE=64
GRADIENT_ACCUMULATION_STEPS=1

# Add validation
if ! command -v accelerate &> /dev/null; then
    echo "Error: accelerate not found. Please install it first."
    exit 1
fi

accelerate launch \
--num_processes=$NUM_PROCESSES \
--num_machines=1 \
--mixed_precision="bf16" \
--dynamo_backend="inductor" \
sft.py \
--config_file configs/sft/script_args.yaml \
--config_file configs/sft/model_config.yaml \
--config_file configs/sft/sft_config.yaml \
--tags example sft test \
--notes "sft test run" \
--output_dir outputs/sft \
--max_steps $MAX_STEPS \
--eval_steps $((MAX_STEPS / 10)) \
--auto_batch_size $AUTO_BATCH_SIZE \
--per_device_train_batch_size $BATCH_SIZE \
--per_device_eval_batch_size $BATCH_SIZE \
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
--push_to_hub true \
--hub_model_id ${HUB_ID}/research-template-example \
--hub_strategy checkpoint \
--hub_private_repo true

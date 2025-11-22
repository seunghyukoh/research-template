#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: bash scripts/sft.sh

set -e

source .env

export HYDRA_RUN_ID=$(date -u +%Y%m%d-%H%M%S)



NUM_PROCESSES=${NUM_PROCESSES:-2}
PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-512}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-512}

accelerate launch --num_processes=${NUM_PROCESSES} run_sft.py \
logging.exp_id=${HYDRA_RUN_ID} \
logging.exp_name="test" \
+logging.tags=["sft","debug"] \
training.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
training.per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}

#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: bash scripts/sft.sh

set -e

source .env

export HYDRA_RUN_ID=$(date -u +%Y%m%d-%H%M%S)



NUM_PROCESSES=${NUM_PROCESSES:-2}
BATCH_SIZE=${BATCH_SIZE:-64}

# Use ceiling division to avoid losing samples
PER_DEVICE_TRAIN_BATCH_SIZE=$(((BATCH_SIZE + NUM_PROCESSES - 1) / NUM_PROCESSES))
PER_DEVICE_EVAL_BATCH_SIZE=$PER_DEVICE_TRAIN_BATCH_SIZE

# Warn if batch size is not evenly divisible
if (( BATCH_SIZE % NUM_PROCESSES != 0 )); then
    echo "Warning: BATCH_SIZE (${BATCH_SIZE}) is not evenly divisible by NUM_PROCESSES (${NUM_PROCESSES})."
    TOTAL_BATCH_SIZE=$((PER_DEVICE_TRAIN_BATCH_SIZE * NUM_PROCESSES))
    echo "Actual total batch size will be ${TOTAL_BATCH_SIZE} (requested ${BATCH_SIZE})."
fi
accelerate launch --num_processes=${NUM_PROCESSES} run_sft.py \
logging.exp_id=${HYDRA_RUN_ID} \
logging.exp_name="test" \
+logging.tags=["sft","debug"] \
training.per_device_train_batch_size=${PER_DEVICE_TRAIN_BATCH_SIZE} \
training.per_device_eval_batch_size=${PER_DEVICE_EVAL_BATCH_SIZE}

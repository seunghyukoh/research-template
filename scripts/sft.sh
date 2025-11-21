#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: bash scripts/sft.sh

set -e

source .env

export HYDRA_RUN_ID=$(date -u +%Y%m%d-%H%M%S)

accelerate launch sft.py \
logging.exp_id=${HYDRA_RUN_ID} \
logging.exp_name="test" \
+logging.tags=["sft","debug"]

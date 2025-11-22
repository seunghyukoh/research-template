#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: bash scripts/sft.sh

set -e

source .env

export HYDRA_RUN_ID=$(date -u +%Y%m%d-%H%M%S)

accelerate launch run_sft.py \
logging.run_id=${HYDRA_RUN_ID} \
logging.run_name="test" \
logging.exp_id="000-demo-sft" \
+logging.tags=["sft","debug"]

#!/bin/bash
# Supervised Fine-Tuning (SFT) Training Script
# Usage: bash scripts/sft.sh

set -e

source .env

accelerate launch sft.py \
logging.exp_name="test" \
+logging.tags=["sft","debug"]

#!/bin/bash

set -eou pipefail

uv run accelerate launch -m lm_eval run \
    --model hf \
    --model_args pretrained=gpt2 \
    --tasks hellaswag

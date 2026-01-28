#!/bin/bash

set -eou pipefail

uv run lm-eval  run --model hf --model_args pretrained=gpt2 --tasks hellaswag

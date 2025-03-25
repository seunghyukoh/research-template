#!/bin/bash
source utils/remote-debug.sh

# Common configuration
WANDB_GROUP="finetune"
DEFAULT_ARGS_FILE="experiments/finetune/args/default.yaml"

# Generate a unique identifier for the experiment
generate_uuid() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        uuidgen
    else
        cat /proc/sys/kernel/random/uuid
    fi
}

# Run fine-tuning with standard configuration
# Usage: finetune [additional arguments]
function finetune {
    uuid=$(generate_uuid)
    python src/finetune.py \
        --id "$uuid" \
        --wandb_group "$WANDB_GROUP" \
        --args_file "$DEFAULT_ARGS_FILE" \
        "$@"
}

# Run fine-tuning in debug mode
# Usage: debug-finetune [additional arguments]
function debug-finetune {
    uuid=$(generate_uuid)
    debug src/finetune.py \
        --id "$uuid" \
        --wandb_group "$WANDB_GROUP" \
        --args_file "$DEFAULT_ARGS_FILE" \
        "$@"
}

# Run distributed fine-tuning
# Usage: finetune-distributed PORT [additional arguments]
function finetune-distributed {
    uuid=$(generate_uuid)
    accelerate launch --main_process_port "$1" src/finetune.py \
        --id "$uuid" \
        --wandb_group "$WANDB_GROUP" \
        --args_file "$DEFAULT_ARGS_FILE" \
        "${@:2}"
}

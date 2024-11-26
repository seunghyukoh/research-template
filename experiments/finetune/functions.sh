#!/bin/bash
source utils/remote-debug.sh

function finetune {
    python src/finetune.py \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        $@
}

function debug-finetune {
    debug src/finetune.py \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        $@
}

function finetune-distributed {
    accelerate launch --main_process_port $1 src/finetune.py \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        ${@:2}
}

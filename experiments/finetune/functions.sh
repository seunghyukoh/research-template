#!/bin/bash
source utils/remote-debug.sh


function finetune {
    uuid=`cat /proc/sys/kernel/random/uuid`
    python src/finetune.py \
        --id $uuid \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        $@
}

function debug-finetune {
    uuid=`cat /proc/sys/kernel/random/uuid`
    debug src/finetune.py \
        --id $uuid \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        $@
}

function finetune-distributed {
    uuid=`cat /proc/sys/kernel/random/uuid`
    accelerate launch --main_process_port $1 src/finetune.py \
        --id $uuid \
        --wandb_group finetune \
        --args_file experiments/finetune/args/default.yaml \
        ${@:2}
}

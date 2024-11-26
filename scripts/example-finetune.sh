source experiments/finetune/functions.sh

WANDB_MODE=disabled finetune \
    --run_name debug \
    --args_file experiments/finetune/args/gsm8k-1epoch-lr1e-4.yaml \
    --model_name_or_path meta-llama/Llama-3.2-3B-Instruct \
    --output_dir output/debug/gsm8k-1epoch-lr1e-4 \
    --loss_type cross_entropy \
    --wandb_tags debug

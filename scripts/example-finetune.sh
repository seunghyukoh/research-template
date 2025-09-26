source experiments/finetune/functions.sh

WANDB_MODE=disabled finetune \
--run_name debug \
--wandb_tags debug gpt2 lr1e-5 epoch1 warmup0.1 \
--args_file experiments/finetune/args/default-chat-template.yaml \
--model_name gpt2 \
--model_path openai-community/gpt2 \
--dataset_name gsm8k \
--dataset_path openai/gsm8k \
--loss_type cross_entropy \
--num_train_epochs 1 \
--learning_rate 1e-5 \
--warmup_ratio 0.1 \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 2

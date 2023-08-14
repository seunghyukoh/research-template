nvidia-smi

base_model=${BASE:-"opt-1.3B"}
bsz=${BATCH:-64} # Batch size per device
lr=${LR:-2e-5}
warmup_steps=${WU:-1000}
save_steps=${SAVE:-1000}
num_gpus=${NUM_GPUS:-4}
num_train_epochs=1


################################

total=$((${bsz} * ${num_gpus}))
echo $total

run_name_suffix="bsz${total}"
run_name="${base_model}_${run_name_suffix}"

echo "Run: ${run_name}"

cache_dir=./.cache
out_dir=checkpoints/$run_name
mkdir -p $out_dir

export WANDB_DIR=$out_dir

header="torchrun
--standalone \
--nodes=1 \
--nproc_per_node=$num_gpus \
src/train.py"

model_url="facebook/${base_model}"

arguments=(
    --report_to wandb
    --config_name $model_url
    --tokenizer_name $model_url
    --model_name_or_path $model_url
    --per_device_eval_batch_size $bsz
    --per_device_train_batch_size $bsz
    --learning_rate $lr
    --warmup_steps $warmup_steps
    --do_train
    --do_eval
    --evaluation_strategy steps
    --logging_steps 1
    --eval_steps $save_steps
    --save_steps $save_steps
    --preprocessing_num_workers 8
    --dataloader_num_workers 8
    --cache_dir $cache_dir
    --add_special_tokens false
    --max_eval_samples 2000
    --num_train_epochs ${num_train_epochs}
    --disable_tqdm true
    --resume_from_checkpoint true
    --log_level info
    --learning_rate $lr
    --run_name $run_name
    --output_dir $out_dir
    --remove_unused_columns false
    $@
)


echo "Training ${base_model} with lr ${lr}, bsz ${bsz} per device"
echo Outputting to $out_dir

echo command: "$header ${arguments[@]}"
$header ${arguments[@]} 2>&1 | tee -a $out_dir/log-resume.out

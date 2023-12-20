# Prepare model
if [ "$5" == "7b" ]; then
    model="mistralai/Mistral-7B-v0.1"
else
    echo "Model $5 not supported"
    exit 1
fi

deepspeed --num_gpus 4 --num_nodes 1 cli/train.py \
    --output_dir .cache/output \
    --init_ckpt $model \
    --data_path .cache/data/$1.jsonl \
    --dry_run \
    --max_seq_len $2 \
    --train_steps 10 \
    --eval_steps 20 \
    --save_steps 100 \
    --log_steps 5 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --use_fused_ops false \
    --deepspeed_config ./configs/$3.json \
    --experiment_name $4 \
    --window_size $6
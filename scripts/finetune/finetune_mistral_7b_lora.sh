
export PYTHONPATH=$(pwd)

deepspeed --num_gpus 2 --num_nodes 1 cli/train.py \
    --output_dir /.cache/models \
    --init_ckpt ./mistral-7b \
    --data_path ./lamini.jsonl \
    --max_seq_len 512 \
    --train_steps 1000 \
    --eval_steps 100 \
    --save_steps 1000 \
    --log_steps 10 \
    --pipe_parallel_size 2 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --use_fused_ops false \
    --deepspeed_config ./configs/mistral_lora.json
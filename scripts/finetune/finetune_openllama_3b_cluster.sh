deepspeed --hostfile scripts/hostfiles/openllama-3b \
    --exclude="sgs-gpu03:1" \
    starter.py \
    --output_dir .cache/models/openllama-3b-chat \
    --init_ckpt /mnt/scratch/xiayao/cache/pretrained_weights/openllama-3b-v2 \
    --data_path /mnt/scratch/xiayao/cache/datasets/massive_dialogs/ar/train.jsonl \
    --max_seq_len 256 \
    --train_steps 20000 \
    --eval_steps 10 \
    --save_steps 1000 \
    --log_steps 1 \
    --pipe_parallel_size 7 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ./configs/llama.json
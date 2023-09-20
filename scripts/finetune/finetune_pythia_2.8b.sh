deepspeed --num_gpus 4 --num_nodes 1 cli/train.py \
    --output_dir .cache/models \
    --init_ckpt /mnt/scratch/xiayao/cache/pretrained_weights/pythia-2.8b-deduped \
    --data_path .cache/data/dialogs.jsonl \
    --max_seq_len 2048 \
    --train_steps 10000 \
    --eval_steps 10 \
    --save_steps 1000 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --use_fused_ops false \
    --deepspeed_config ./configs/pythia.json
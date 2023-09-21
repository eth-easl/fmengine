torchrun --nnodes 1 --nproc-per-node 4 --node_rank 0 \
 cli/train.py \
    --output_dir /workspace/.cache/models \
    --init_ckpt /pretrained/open_llama_3b_v2 \
    --data_path /datasets/prompt.jsonl \
    --max_seq_len 2048 \
    --train_steps 10000 \
    --eval_steps 10 \
    --save_steps 250 \
    --log_steps 1 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --use_fused_ops false \
    --deepspeed_config ./configs/llama_lessmem.json
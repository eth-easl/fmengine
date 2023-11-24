CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes 1 --nproc-per-node 4 --node_rank 0 \
 cli/train.py \
    --output_dir /workspace/.cache/models-7b \
    --init_ckpt /pretrained/llama-2-7b-mp2 \
    --data_path /datasets/data.jsonl \
    --max_seq_len 2048 \
    --train_steps 10000 \
    --eval_steps 10 \
    --save_steps 1000 \
    --log_steps 1 \
    --pipe_parallel_size 2 \
    --model_parallel_size 2 \
    --use_flash_attn true \
    --use_fused_ops true \
    --deepspeed_config ./configs/llama_lora.json
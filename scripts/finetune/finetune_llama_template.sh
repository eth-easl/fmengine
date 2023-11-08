# $1: dataset demo or dialog
# $2: max_seq_len
# $3: config llama_lora or llama
# $4: exp_name: <date>-<dataset>-<config>-<max_seq_len>

# TODO: use default options

deepspeed --num_gpus 4 --num_nodes 1 cli/train.py \
    --output_dir .cache/models \
    --init_ckpt openlm-research/open_llama_3b_v2 \
    --data_path .cache/data/$1.jsonl \
    --max_seq_len $2 \
    --train_steps 10 \
    --eval_steps 20 \
    --save_steps 100 \
    --log_steps 5 \
    --pipe_parallel_size 4 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --deepspeed_config ./configs/$3.json \
    --exp_name $4
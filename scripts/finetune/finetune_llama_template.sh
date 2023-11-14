# $1: dataset demo or dialog
# $2: max_seq_len
# $3: config llama_lora or llama
# $4: exp_name: <date>-<dataset>-<config>-<max_seq_len>
# $5: model: .cache/model/llama-2-7b-hf or openlm-research/open_llama_3b_v2
# $6: window_size: 256

# Prepare model
if [ "$5" == "7b" ]; then
    model=".cache/model/llama-2-7b-hf"
elif [ "$5" == "3b" ]; then
    model="openlm-research/open_llama_3b_v2"
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
    --use_fused_ops true \
    --deepspeed_config ./configs/$3.json \
    --exp_name $4 \
    --window_size $6
# Distributed Training

## Multi-GPU training

Running multi-GPU training is as simple as running the following command:

```bash
deepspeed --num_gpus 2 --num_nodes 1 cli/train.py \
    --output_dir /workspace/.cache/models \
    --init_ckpt /pretrained/tinyllama-2-1b \
    --data_path /datasets/dataset.train.jsonl \
    --max_seq_len 128 \
    --train_steps 1000 \
    --eval_steps 10 \
    --save_steps 1000 \
    --log_steps 10 \
    --pipe_parallel_size 2 \
    --model_parallel_size 1 \
    --use_flash_attn true \
    --use_fused_ops false \
    --deepspeed_config ./configs/llama.json
``` 

## Multi-host training

FMEngine support multi-host training with deepspeed. To run multi-host training, you need to install [`pdsh`](https://github.com/chaos/pdsh) first, by running the following command:

```bash
git clone https://github.com/chaos/pdsh.git
cd pdsh
./configure --enable-static-modules --without-rsh --with-ssh --without-ssh-connect-timeout-option --prefix=/your/preferred/path
make
make install
```

## Slurm and HPC

On slurm clusters, you can use the following script to run FMEngine:

```bash
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
singularity run --nv \
--home <path_to_your_home>:/home/<your_username> \
--bind <path_to_your_HF_HOME>:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind <path_to_your_pretrained_weights>:/pretrained \
--bind <path_to_your_datasets> \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
bash scripts/finetune/finetune_llama_1b_local.sh
```

It is strongly recommended to use singlarity/apptainer to run FMEngine on HPC clusters.
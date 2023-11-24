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

It is strongly recommended to use singlarity/apptainer to run FMEngine on HPC clusters. However, the `deepspeed` launcher cannot trigger slurm command within singularity container (It's probably doable, but quite complicated). Here we suggest to start the training script with `torchrun` instead.

Before running training scripts, you need to login to your wandb account first, by running the following command:

```bash

$ singularity shell fmsys.sif
> wandb login
> # Enter your API key
```

The wandb configuration will be stored in your home directory (it is supposed to be the same directory on your HPC and within the singularity container), therefore you will need to mount your home directory to the singularity container.

Here's an example script of running FMEngine on slurm clusters.

```bash
#!/bin/bash -x
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --partition=<YOUR_PARTITION>
#SBATCH --output=/<YOUR_LOG_PATH>/%j_%N_log.out

# Network Configuration, cluster-specific
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_ASYNC_ERROR_HANDLING=1

echo $SLURM_JOB_GPUS
echo $SLURM_NTASKS
echo $SLURM_NODELIST

# Convert SLURM_JOB_GPUS to an array
IFS=',' read -ra GPU_ARRAY <<< "$SLURM_JOB_GPUS"

# Get the number of GPUs from the length of the array
NUM_GPUS=${#GPU_ARRAY[@]}

export TOTAL_GPUS=$(($NUM_GPUS * $SLURM_NTASKS))
echo $TOTAL_GPUS

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"

export MASTER_ADDR=$master_addr

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

# Print System Information
echo "GPUs available to job: $SLURM_JOB_GPUS"
echo "Total tasks: $SLURM_NTASKS"

# Loop over all nodes
for ((i=0; i<$COUNT_NODE; i++))
do
    srun --nodes=1 --ntasks=1 -w "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "$((i+1))p")" \
    singularity run --nv \
    --home <YOUR_HOME_DIRECTORY>:/home/<YOUR_USERNAME> \
    --bind <YOUR_CACHE_DIRECTORY>/transformers_cache:/.hf_cache \
    --env HF_HOME=/.hf_cache \
    --env PYTHONPATH=/workspace \
    --bind <YOUR_CACHE_DIRECTORY>/pretrained_weights:/pretrained \
    --bind <YOUR_CACHE_DIRECTORY>/datasets:/datasets \
    --bind $PWD:/workspace \
    --pwd /workspace \
    fmsys_0.0.4.sif \
    torchrun \
    --master_addr "$MASTER_ADDR" \
    --master_port 12802 \
    --node_rank $i \
    --nnodes $SLURM_NTASKS \
    --nproc-per-node=$NUM_GPUS \
        cli/train.py \
        --output_dir /workspace/.cache/models \
        --init_ckpt /pretrained/llama-2-7b-hf \
        --data_path /datasets/prompt.jsonl \
        --max_seq_len 2048 \
        --train_steps 10 \
        --eval_steps 10 \
        --save_steps 1000 \
        --log_steps 1 \
        --pipe_parallel_size 8 \
        --model_parallel_size 1 \
        --use_flash_attn true \
        --use_fused_ops false \
        --deepspeed_config ./configs/llama.json &
done

wait
```
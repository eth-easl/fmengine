#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
singularity run --nv \
--home /p/home/jusers/yao4/juwels/:/home/xiayao \
--bind /p/home/jusers/yao4/juwels/shared/fmsys/cache/HF:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind /p/home/jusers/yao4/juwels/shared/fmsys/cache/pretrained_weights:/pretrained \
--bind /p/home/jusers/yao4/juwels/shared/fmsys/cache/datasets:/datasets \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
bash scripts/finetune/finetune_llama_1b_local.sh
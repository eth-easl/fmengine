singularity run --nv \
--home /mnt/scratch/xiayao:/home/xiayao \
--bind /mnt/scratch/xiayao/cache/HF/hub:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind /mnt/scratch/xiayao/cache/pretrained_weights:/pretrained \
--bind /mnt/scratch/xiayao/cache/datasets:/datasets \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
bash scripts/finetune/finetune_llama_1b_torchrun.sh
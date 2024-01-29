singularity shell --nv \
--home /mnt/scratch/xiayao:/home/xiayao \
--bind /mnt/scratch/xiayao/cache/HF/:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind .cache/ckpts:/pretrained \
--bind .cache/data:/datasets \
--bind $PWD:/workspace \
--pwd /workspace \
fmengine.sif
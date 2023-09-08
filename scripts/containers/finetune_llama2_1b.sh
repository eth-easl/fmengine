singularity run --nv \
--bind .:/home/xiayao/ \
--bind /mnt/scratch/xiayao/cache/pretrained_weights:/pretrained \
--bind /mnt/scratch/xiayao/cache/datasets:/datasets \
fmsys.sif \
bash scripts/finetune/finetune_llama_1b_local.sh
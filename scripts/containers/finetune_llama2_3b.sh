singularity run --nv \
--home /mnt/scratch/xiayao:/home/xiayao \
--bind /mnt/scratch/xiayao/cache/HF/hub:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind /mnt/scratch/xiayao/cache/pretrained_weights:/pretrained \
--bind .cache/data/dialogs.jsonl:/datasets/data.jsonl \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
bash scripts/finetune/finetune_llama_3b.sh
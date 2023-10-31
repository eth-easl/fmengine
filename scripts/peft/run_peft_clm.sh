singularity run --nv \
--home /mnt/scratch/xiayao:/home/xiayao \
--bind /mnt/scratch/xiayao/cache/HF/hub:/.hf_cache \
--env HF_HOME=/.hf_cache \
--env PYTHONPATH=/workspace/ \
--bind /mnt/scratch/xiayao/cache/pretrained_weights:/pretrained \
--bind .cache/data/dialogs.jsonl:/datasets/data.jsonl \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys.sif \
python cli/peft_for_clm.py --model-name-or-path openlm-research/open_llama_3b_v2 --lora-rank 1 --data-path .cache/data/sqlg.train.ar.jsonl --wandb-run-name openllama-3b-sqlg-r8 --project-name ft-research --max-seq-len 512
singularity shell --nv \
--home /home/xzyao:/home/xiayao \
--bind /home/xzyao/.cache/huggingface/hub:/.hf_cache \
--env HF_HOME=/.hf_cache \
--bind .cache/ckpts:/pretrained \
--bind .cache/data:/datasets \
--bind $PWD:/workspace \
--pwd /workspace \
fmsys_0.0.10.sif
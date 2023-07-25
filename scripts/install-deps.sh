# torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# Utilities
pip install scikit-learn numpy pandas loguru
# HF
pip install deepspeed transformers datasets accelerate tokenizers evaluate
# nvcc and other cuda toolkits - required for flash attention
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
# flash attention
pip install packaging ninja flash-attn
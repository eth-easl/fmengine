# torch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
# Utilities
pip install -r requirements.txt
# nvcc and other cuda toolkits - required for flash attention
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
# flash attention
pip install packaging ninja flash-attn
# mpi4py
conda install -c conda-forge mpi4py
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
mkdir build
cd build
git clone https://github.com/NVIDIA/apex apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
cd ../..
rm -rf build
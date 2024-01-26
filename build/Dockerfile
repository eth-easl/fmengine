FROM nvcr.io/nvidia/pytorch:23.10-py3

ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=$CUDA_HOME/include:$CPATH
ENV MAX_JOBS=1
ARG TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX;8.9;9.0"

COPY requirements.txt /env/requirements.txt
RUN apt update && apt upgrade -y
RUN pip install --upgrade pip && pip install -r /env/requirements.txt
# RUN git clone https://github.com/NVIDIA/apex apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && cd .. && rm -rf apex
# since cufile is provided by kvikio, we remove the one from the base image to build kvikio, then put it back
RUN pip install flash-attn --no-build-isolation
RUN pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/rotary && \
    pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/layer_norm && \ 
    pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/xentropy && \
    pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/fused_dense_lib && \
    pip install git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=csrc/fused_softmax

RUN mv /usr/local/cuda/include/cufile.h /tmp/cufile.h && git clone --branch branch-23.10 https://github.com/rapidsai/kvikio.git && cd kvikio && ./build.sh kvikio && cd .. && rm -rf kvikio && mv /tmp/cufile.h /usr/local/cuda/include/cufile.h
RUN pip install -U xformers
RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
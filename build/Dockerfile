FROM nvcr.io/nvidia/pytorch:23.08-py3
ENV CUDA_HOME=/usr/local/cuda
ENV CPATH=$CUDA_HOME/include:$CPATH
COPY requirements.txt /env/requirements.txt
RUN apt update && apt upgrade -y
RUN pip install --upgrade pip && pip install -r /env/requirements.txt
RUN pip install flash-attn --no-build-isolation
# check how many files are in CUDA_HOME
RUN git clone https://github.com/NVIDIA/apex apex && cd apex && pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ && cd .. && rm -rf apex
# since cufile is provided by kvikio, we remove the one from the base image to build kvikio, then put it back
RUN mv /usr/local/cuda/include/cufile.h /tmp/cufile.h && git clone --branch branch-23.10 https://github.com/rapidsai/kvikio.git && cd kvikio && ./build.sh kvikio && cd .. && rm -rf kvikio && mv /tmp/cufile.h /usr/local/cuda/include/cufile.h
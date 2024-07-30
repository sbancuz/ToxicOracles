FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
LABEL description="Llama-CPP-Python server (docs: https://github.com/abetlen/llama-cpp-python)."
# Setup environment
## Install required tools
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
    python3 python3-pip gcc wget \
    ocl-icd-opencl-dev opencl-headers clinfo \
    libclblast-dev libopenblas-dev \
    && mkdir -p /etc/OpenCL/vendors && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
## Create directories
RUN mkdir /app
RUN mkdir /app/llm
RUN mkdir /app/llm/resources/
RUN mkdir /app/llm/resources/models
## Declare volumes
VOLUME /app/llm/resources/models/pre_trained
## Set current working directory
WORKDIR /app/llm
# Install Python tools
## Setup pip
RUN python3 -m pip install --no-cache-dir --upgrade pip
## Install depencencies
# TODO find better way to specify versions
RUN python3 -m pip install --upgrade pip pytest==8.2.2 cmake==3.29.5 scikit-build==0.17.6 setuptools fastapi==0.111.0 uvicorn==0.30.1 sse-starlette==2.1.0 pydantic-settings==2.3.2 starlette-context==0.3.6
## Install llama-cpp-python (building with cuda)
RUN CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.78
# Set host to allow external access
ENV HOST 0.0.0.0
# Port mappig
EXPOSE 8000

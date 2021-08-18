# This Dockerfile will build Docker Image with 
#    ONNX (from ONNXRuntime submodule) + ONNXRuntime + ONNXRuntime-Extensions
#    installed for CPU only
#
# Example commandline to build this full package set.
# docker build . -t docker-image-repo-name
# Example commandline to run the built docker container:
# sudo docker run --name container-name -it docker-image-repo-name

ARG PYTHON_VERSION=3.8

FROM ubuntu:18.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils       \
        build-essential \
        ca-certificates \
        ccache  \
        cmake   \
        curl    \
        git     \
        wget    \
        gcc-8   \
        g++-8   \
        libprotobuf-dev \
        protobuf-compiler && \
    rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-8 100
RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

ENV PATH /opt/conda/bin:$PATH

RUN wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION} conda-build && \
    /opt/conda/bin/conda install -y numpy ninja setuptools cmake protobuf typing-extensions && \
    /opt/conda/bin/conda clean -ya

ADD . /source
WORKDIR /root
RUN if [ -d /source/onnnxruntime ] ; then ln -s /source/onnxruntime /root/onnxruntime ; fi
RUN if [ ! -L /root/onnxruntime ] ; then git clone https://github.com/microsoft/onnxruntime.git && \
    cd onnxruntime && git submodule update --init --recursive ; fi

RUN cd /root/onnxruntime/cmake/external/onnx && \
    CMAKE_ARGS="-DONNX_USE_PROTOBUF_SHARED_LIBS=ON" python3 setup.py bdist_wheel && \
    python3 -m pip install dist/*.whl

RUN cd /root/onnxruntime && \
    /bin/bash ./build.sh    \
    --config Release \
    --build_wheel   \
    --update        \
    --build         \
    --parallel      \
    --skip_submodule_sync \
    --cmake_extra_defines \
    ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) && \
    python3 -m pip install /root/onnxruntime/build/Linux/Release/dist/*.whl

RUN cd /root/onnxruntime/cmake/external/onnxruntime-extensions && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/*.whl

RUN if [ -L /root/onnxruntime ]; then unlink /root/onnxruntime ; fi && \
    if [ -d /root/onnxruntime ]; then rm -rf /root/onnxruntime ; fi && \
    rm -rf /opt/ccache

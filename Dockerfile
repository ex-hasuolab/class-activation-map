FROM tensorflow/tensorflow:2.2.0-gpu

ARG DEBIAN_FRONTEND=noninteractive

# Install apt dependencies
RUN apt-get update && apt-get install -y \
    git \
    gpg-agent \
    python3-cairocffi \
    protobuf-compiler \
    python3-pil \
    python3-lxml \
    python3-tk \
    wget

# Install gcloud and gsutil commands
# https://cloud.google.com/sdk/docs/quickstart-debian-ubuntu
RUN export CLOUD_SDK_REPO="cloud-sdk-$(lsb_release -c -s)" && \
    echo "deb http://packages.cloud.google.com/apt $CLOUD_SDK_REPO main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# tensorflowユーザーの追加(-msでhomeディレクトリを作成)
RUN useradd -ms /bin/bash tensorflow
RUN usermod -u 1000 tensorflow
RUN groupmod -g 1000 tensorflow

RUN echo "root:root" | chpasswd && \
    echo "${username}:${username}" | chpasswd

USER tensorflow
RUN python -m pip install --user jupyterlab

# Compile protobuf configs
# Copy this version of of the model garden into the image
# COPY --chown=tensorflow . /home/tensorflow/models
WORKDIR /home/tensorflow
RUN git clone --depth 1 https://github.com/tensorflow/models && chown -R tensorflow models/
RUN (cd /home/tensorflow/models/research/ && protoc object_detection/protos/*.proto --python_out=.)
WORKDIR /home/tensorflow/models/research/
ENV PATH="/home/tensorflow/.local/bin:${PATH}"
# 上でpip installされるtensorflow:2.3.0を削除
RUN python -m pip install -U pip
RUN cp object_detection/packages/tf2/setup.py ./
RUN python -m pip install .
RUN python -m pip uninstall -y tensorflow
RUN python -m pip install jupyterlab imageio scikit-learn keras

WORKDIR /home/tensorflow/workspace
RUN git clone https://github.com/ex-hasuolab/deeplearning-set.git

ENV TF_CPP_MIN_LOG_LEVEL 3
RUN jupyter notebook --generate-config
RUN echo "c.NotebookApp.token = ''" >> ~/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.notebook_dir = '/home/tensorflow/" >> ~/.jupyter/jupyter_notebook_config.py
CMD jupyter lab --port 8888 --ip=0.0.0.0

# Confirmed on ABCI Singluarity
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime
ARG PYTHON_VERSION=3.7

RUN apt-get update
RUN apt-get install -y wget git make tmux

RUN pip install -U pip
RUN conda install -y h5py=2.8.0 \
 && conda clean -ya
RUN pip install h5py-cache==1.0

RUN pip install scipy

RUN pip install --no-index torch-scatter -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-index torch-sparse -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-index torch-cluster -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install --no-index torch-spline-conv -f https://data.pyg.org/whl/torch-1.9.0+cu111.html && \
    pip install torch-geometric==2.0.3 pytorch-lightning torchmetrics


RUN pip install tensorboard matplotlib flake8 yapf optuna ogb



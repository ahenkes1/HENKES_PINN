# Base image
FROM tensorflow/tensorflow:latest-gpu
# # Requirements
# ADD requirements.txt .
# RUN pip install -r requirements.txt


# Arguments
ARG uid=1000
ARG gid=1000
ARG requirements=requirements.txt
ARG entrypoint=docker-entrypoint.sh

# Change default shell
SHELL ["/bin/bash", "--login", "-c"]

# Create a non-root user
ENV USER_NAME docker_user
ENV USER_UID $uid
ENV USER_GID $gid
ENV HOME_DIR /home/$USER_NAME

RUN groupadd --gid $USER_GID $USER_NAME

RUN adduser \
    --disabled-password \
    --gecos "non-root user" \
    --uid $USER_UID \
    --gid $USER_GID \
    --home $HOME_DIR \
    $USER_NAME

# Copy files
# Requirements
COPY $requirements /tmp/
# Add source files and save directory
RUN mkdir -p $HOME_DIR/src/ && mkdir -p $HOME_DIR/src/saved_nets/CPINN/
COPY src/* $HOME_DIR/src/

RUN chown $USER_UID:$USER_GID /tmp/$requirements
RUN chown -R $USER_UID:$USER_GID $HOME_DIR/src

### Install base utilities
RUN apt-get update && \
    apt-get install -y vim && \
    apt-get install -y htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Requirements
RUN pip install --upgrade pip
RUN pip install -r /tmp/$requirements

# Set user
USER $USER_NAME
WORKDIR $HOME_DIR/src

# CMD
ENV XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-11.2 
ENTRYPOINT ["python3", "/home/docker_user/src/main.py"]

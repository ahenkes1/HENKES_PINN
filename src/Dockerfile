# Base image
FROM tensorflow/tensorflow:latest-gpu
# Requirements
ADD requirements.txt .
RUN pip install -r requirements.txt

# Arguments
# ARG uid=1000
# ARG gid=1000
# ARG requirements=requirements.txt
# ARG entrypoint=docker-entrypoint.sh

### Change default shell
# SHELL ["/bin/bash", "--login", "-c"]

### Create a non-root user
# For more information, see:
# - https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
# - https://vsupalov.com/docker-shared-permissions/
# ENV USER_NAME docker_user
# ENV USER_UID $uid
# ENV USER_GID $gid
# ENV HOME_DIR /home/$USER_NAME

# RUN groupadd --gid $USER_GID $USER_NAME

# RUN adduser \
#     --disabled-password \
#     --gecos "non-root user" \
#     --uid $USER_UID \
#     --gid $USER_GID \
#     --home $HOME_DIR \
#     $USER_NAME

### Copy files
# Requirements
# COPY $requirements /tmp/
# RUN chown $USER_UID:$USER_GID /tmp/$requirements
# 
# ### Install base utilities
# RUN apt-get update && \
#     apt-get install -y build-essential  && \
#     apt-get install -y wget && \
#     apt-get install -y nano && \
#     apt-get install -y vim && \
#     apt-get install -y htop && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*
# 
# # Requirements
# RUN pip install --upgrade pip
# RUN pip install -r /tmp/$requirements
# 
# ### Create working directory
# ENV APP_DIR $HOME_DIR/app
# RUN mkdir -p $APP_DIR
# 
# # Set user
# USER $USER_NAME
# WORKDIR $APP_DIR


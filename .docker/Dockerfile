ARG VARIANT="24.04"
FROM ubuntu:${VARIANT} AS base

# disable prompting during package installation
ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_ROOT_USER_ACTION=ignore

RUN apt-get update && apt-get -y upgrade \
  && apt-get install -y --no-install-recommends \
    git \
    ssh \
    sshfs \
    openssh-server \
    openssh-client \
    axel \
    g++ \
    make \
    ca-certificates \
    vim \
    bash-completion \
    ffmpeg \
    libsm6 \
    libxext6 \
    nvtop \
    && rm -rf /var/lib/apt/lists/*
# install miniconda
ARG CONDA_INSTALL_PREFIX="/root/.conda"
ENV CONDA_INSTALL_PREFIX=${CONDA_INSTALL_PREFIX}
ENV PATH="${CONDA_INSTALL_PREFIX}/bin:${PATH}"
ARG PATH="${CONDA_INSTALL_PREFIX}/bin:${PATH}"

# Install Miniconda with Python 3.12
RUN axel -q -n $(nproc) https://repo.anaconda.com/miniconda/Miniconda3-py312_24.9.2-0-Linux-x86_64.sh -o /root/miniconda_install.sh \
  && bash /root/miniconda_install.sh -b -p ${CONDA_INSTALL_PREFIX} \
  && rm /root/miniconda_install.sh \
  && echo "Running $(conda --version)"

ENV PYTHON_INSTALL_LOC=${CONDA_INSTALL_PREFIX}/envs/hirisenet/bin/python

# Setup SSH
RUN mkdir -p -m 0700 ${HOME}/.ssh && \
    mkdir /var/run/sshd && \
    echo "StrictHostKeyChecking no " > ~/.ssh/config && \
    ssh-keyscan github.com >> ~/.ssh/known_hosts

# Create symbolic links to the secrets so that we can use them with git
# Note that we need to delete any existing files to avoid a simlink error
RUN ln -s /run/secrets/ssh_public_key ~/.ssh/id_ed25519.pub && \
   ln -s /run/secrets/ssh_private_key ~/.ssh/id_ed25519 && \
   ln -s /run/secrets/gitconfig ~/.gitconfig

# Use NVIDIA CUDA as base image and run the same installation as in the other packages.
# The version of cudatoolkit must match those of the base image, see Dockerfile.pytorch
#FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
FROM nvcr.io/nvidia/nvhpc:21.2-runtime-cuda11.2-ubuntu20.04

# This is a concatenated Dockerfile, the maintainers of subsequent sections may vary.
RUN chmod 1777 /tmp && chmod 1777 /var/tmp

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"

# Fix DL4006
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

USER root

# Install all OS dependencies for notebook server that starts but lacks all
# features (e.g., download as all possible file formats)
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
    wget \
    bzip2 \
    ca-certificates \
    sudo \
    locales \
    fonts-liberation \
    run-one \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -yq --no-install-recommends \
    build-essential \
    emacs-nox \
    vim-tiny \
    git \
    inkscape \
    jed \
    openssh-client \
    less \
    rsync \
    libsm6 \
    libxext-dev \
    libxrender1 \
    lmodern \
    netcat \
    python-dev \
    tmux \
    # ---- nbconvert dependencies ----
    #texlive-xetex \
    #texlive-fonts-recommended \
    #texlive-plain-generic \
    # Optional dependency
    #texlive-fonts-extra \
    # ----
    tzdata \
    unzip \
    nano \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN apt-get update && sudo apt-get install -y apt-transport-https gnupg2 curl && \
    curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | tee -a /etc/apt/sources.list.d/kubernetes.list && \
    apt-get update && \
    apt-get install -y kubectl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen

# Configure environment
ENV CONDA_DIR=/opt/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8
ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

# Copy a script that we will use to correct permissions after running certain commands
COPY fix-permissions /usr/local/bin/fix-permissions
RUN chmod a+rx /usr/local/bin/fix-permissions

# Enable prompt color in the skeleton .bashrc before creating the default NB_USER
RUN sed -i 's/^#force_color_prompt=yes/force_color_prompt=yes/' /etc/skel/.bashrc

# Create NB_USER wtih name jovyan user with UID=1000 and in the 'users' group
# and make sure these dirs are writable by the `users` group.
RUN echo "auth requisite pam_deny.so" >> /etc/pam.d/su && \
    sed -i.bak -e 's/^%admin/#%admin/' /etc/sudoers && \
    sed -i.bak -e 's/^%sudo/#%sudo/' /etc/sudoers && \
    useradd -m -s /bin/bash -N -u $NB_UID $NB_USER && \
    mkdir -p $CONDA_DIR && \
    chown $NB_USER:$NB_GID $CONDA_DIR && \
    chmod g+w /etc/passwd && \
    fix-permissions $HOME && \
    fix-permissions $CONDA_DIR

USER $NB_UID
WORKDIR $HOME
ARG PYTHON_VERSION=default

# Setup work directory for backward-compatibility
RUN mkdir /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

# Install conda as jovyan and check the md5 sum provided on the download site
ENV MINICONDA_VERSION=4.8.2 \
    MINICONDA_MD5=87e77f097f6ebb5127c77662dfc3165e \
    CONDA_VERSION=4.8.2

WORKDIR /tmp
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    echo "${MINICONDA_MD5} *Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh" | md5sum -c - && \
    /bin/bash Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    #echo "conda ${CONDA_VERSION}" >> $CONDA_DIR/conda-meta/pinned && \
    conda config --system --prepend channels conda-forge && \
    #conda config --system --set auto_update_conda false && \
    conda config --system --set show_channel_urls true && \
    #conda config --system --set channel_priority strict && \
    #if [ ! $PYTHON_VERSION = 'default' ]; then conda install --yes python=$PYTHON_VERSION; fi && \
    #conda list python | grep '^python ' | tr -s ' ' | cut -d '.' -f 1,2 | sed 's/$/.*/' >> $CONDA_DIR/conda-meta/pinned && \
    conda install --quiet --yes conda pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install Tini
RUN conda install --quiet --yes 'tini=0.18.0' && \
    #conda list tini | grep tini | tr -s ' ' | cut -d ' ' -f 1,2 >> $CONDA_DIR/conda-meta/pinned && \
    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

# Install bert fine-tune prerequisites
RUN conda install --quiet --yes -c conda-forge cudatoolkit-dev=11.3
RUN conda install --quiet --yes pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN conda install --quiet --yes tqdm
RUN conda install --quiet --yes tensorboardx
RUN conda install --quiet --yes -c conda-forge scikit-learn

USER root
RUN fix-permissions /home/$NB_USER

# add jovyan to sudoers list
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers
#RUN echo "$NB_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/notebook
WORKDIR $HOME

# add ytt and genkubejob script
RUN curl -s https://api.github.com/repositories/173207060/releases/latest | grep browser_download_url.*ytt-linux-amd64 | cut -d '"' -f 4 | xargs wget -P /usr/local/bin/
RUN mv /usr/local/bin/ytt-linux-amd64 /usr/local/bin/ytt
RUN chmod +x /usr/local/bin/ytt
ADD genkubejob /usr/local/bin/

# Switch back to jovyan to avoid accidental container runs as root
USER $NB_UID


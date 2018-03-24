# Start with CUDA base image
FROM nvidia/cuda
MAINTAINER John Chodera <john.chodera@choderalab.org>

# Install miniconda
RUN apt-get update && apt-get install -y wget
RUN MINICONDA="Miniconda3-latest-Linux-x86_64.sh" && \
    wget --quiet https://repo.continuum.io/miniconda/$MINICONDA && \
    bash $MINICONDA -b -p /miniconda && \
    rm -f $MINICONDA
ENV PATH /miniconda/bin:$PATH

# Add omnia
RUN conda config --add channels omnia

# Install yank
RUN conda install --yes yank

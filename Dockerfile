# Start with CUDA base image
FROM kaixhin/cuda
MAINTAINER John Chodera <john.chodera@choderalab.org>

# Install curl and dependencies for FAH
RUN apt-get update && apt-get install -y \
  wget

# Install miniconda
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh && \
    bash Miniconda2-latest-Linux-x86_64.sh -b -p /miniconda && \
    rm -f Miniconda2-latest-Linux-x86_64.sh
ENV PATH /miniconda/bin:$PATH

# Add omnia
RUN conda config --add channels omnia

# Install yank
RUN conda install --yes yank

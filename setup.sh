#!/bin/bash

# define conda environment name
CONDA_ENV_NAME=fastml4jets

# install miniforge locally if it doesn't already exist
CONDA_INSTALL=$PWD/conda/

if [[ ! -d "${CONDA_INSTALL}" ]]; then
  CONDA_REPOSITORY=https://github.com/conda-forge/miniforge/releases/latest/download
  # installation for macOS (including support for M1 MacBooks)
  if [[ $OSTYPE == 'darwin'* ]]; then
    MAC_TYPE="$(uname -m)"
    if [[ $MAC_TYPE == 'arm64' ]]; then
      CONDA_INSTALLER="Miniforge3-MacOSX-arm64.sh"
    else
      CONDA_INSTALLER="Miniforge3-MacOSX-x86_64.sh"
    fi
  # installation for linux
  elif [[ $OSTYPE == 'linux'* ]]; then
    CONDA_INSTALLER="Miniforge3-Linux-x86_64.sh"
  # other operating system not supported
  else
    echo "Operating system not supported. Setup not possible."
    exit 1
  fi
  # install miniforge to local directory
  curl -L -O ${CONDA_REPOSITORY}/${CONDA_INSTALLER}
  bash ${CONDA_INSTALLER} -b -p ${CONDA_INSTALL}
  rm ${CONDA_INSTALLER}
fi

# activate conda
source ${CONDA_INSTALL}/bin/activate
# create conda environment
conda install -c conda-forge root
conda env create -f requirements.yaml
conda activate $CONDA_ENV_NAME
# always export the framework paths
export PATH=${CONDA_INSTALL}/envs/${CONDA_ENV_NAME}/bin:$PATH
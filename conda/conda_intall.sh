#!/bin/bash

USAGE="
Usage: $(basename $0)
Options:
-n|--name                Name of the new environment to be created
-b|--build-mpi           Downloads and compiles mpi4py from tarball into provided directory
-r|--build-mpi-from-repo Downloads the latest from the mpi4py repo into provided directory
-g|--gpu                 Bypasses checks for cuda/cudnn and installs tensorflow-gpu
-c|--cpu                 Bypasses checks for cuda/cudnn and installs tensorflow-gpu
-e|--env                 Pass an conda environment.yml file
-h|--help                Displays this message"

ENV="environment.yml"
MPI_REPO="https://github.com/mpi4py/mpi4py.git"
MPI_TAR="https://github.com/mpi4py/mpi4py/archive/3.0.3.tar.gz"

#Parse some arguments
PARAMS=""
while (( "$#" )); do
  case "$1" in
    -n|--name)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        CONDA_ENV_NAME=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -b|--build-mpi)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        BUILD_MPI=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -r|--build-mpi-from-repo)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        BUILD_MPI=$2
        USE_REPO=1
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;

    -g|--gpu)
      BUILD_GPU=1
      shift
      ;;
    -c|--cpu-only)
      BUILD_CPU=1
      shift
      ;;
    -e|--env)
      if [ -n "$2" ] && [ ${2:0:1} != "-" ]; then
        ENV=$2
        shift 2
      else
        echo "Error: Argument for $1 is missing" >&2
        exit 1
      fi
      ;;
    -h|--help)
      echo "$USAGE"
      exit 0
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      echo "$USAGE"
      exit 1
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done

# set positional arguments in their proper place
eval set -- "$PARAMS"
set -e

#Check to make sure user did specify both cpu and gpu flags
if [ -n "$BUILD_GPU" ] && [ -n "$BUILD_CPU" ]; then
  echo 'Cannot set both cpu-only and gpu flags'
  exit 1
fi

#Look for nvcc and for cudnn.h which are required for tensorflow-gpu
if [ -z "$BUILD_CPU" ] && command -v nvcc &> /dev/null; then
    NVCC_PATH=$(which nvcc)
    CUDA_BIN_PATH=$(dirname "$NVCC_PATH")
    CUDA_PATH=$(dirname "$CUDA_BIN_PATH")
    CUDNN_FILE="${CUDA_PATH}/include/cudnn.h"
    echo "$NVCC_PATH $CUDA_BIN_PATH $CUDA_PATH $CUDNN_FILE"
    if test -f "$CUDNN_FILE"; then
      echo "$CUDNN_FILE"
      CUDNN_MAJOR=$(grep -m 1 CUDNN_MAJOR $CUDNN_FILE | awk '{ print $3}')
      CUDNN_MINOR=$(grep -m 1 CUDNN_MINOR $CUDNN_FILE | awk '{ print $3}')
      CUDNN_PATCH=$(grep -m 1 CUDNN_PATCH $CUDNN_FILE | awk '{ print $3}')
      CUDNN_VERSION="${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCH}"
    else
      echo "Could not find cudnn.h in ${CUDA_PATH}/include"
    fi
    echo "NVCC=$NVCC_PATH"
    echo "CUDNN_VERSION=$CUDNN_VERSION"
fi

#Looks at the options and the results of the cuda/cudnn search
#to set if we use tensorflow or tensorflow-gpu
if [ -n "$BUILD_GPU" ] && [ -z "$CUDNN_VERSION" ]; then
  echo "Could not find cudnn.h"
  exit 1
elif [ -n "$BUILD_CPU" ] || [ -z "$CUDNN_VERSION" ]; then
  echo "Building cpu-only"
  sed 's/tensorflow-gpu/tensorflow/g' $ENV > "exaRL_evn_to_build.yml"
else
  echo "Building gpu"
  cp $ENV "exaRL_evn_to_build.yml"
fi

#We are making a copy of the environment because we will be modifying it
#Attempted to use jinja to avoid this, but it seems that it is not
#Supported in conda env create --enviornment.yml yet
ENV="exaRL_evn_to_build.yml"

#Checks if we want to build mpi4py and switches between tarball or repo
#We modify the environment to build them after
if [ -n "$BUILD_MPI" ]; then
  pushd .
  if [ -d $BUILD_MPI ]; then
    echo "Mpi4py dir already exists exiting..."
    rm -rf "$BUILD_MPI"
  fi
  mkdir $BUILD_MPI
  cd $BUILD_MPI
  if [ -z "$USE_REPO" ]; then
    echo "Downloading mpi4py tar"
    wget $MPI_TAR
    TAR_NAME=$(basename $MPI_TAR)
    tar -xzf $TAR_NAME
    MPI_BUILD_PATH="$(pwd)/mpi4pi-3.0.3"
  else
    echo "Downloading mpi4py git"
    git clone $MPI_REPO
    MPI_BUILD_PATH="$(pwd)/mpi4py"
  fi
  popd
  sed -i 's/- pip://' $ENV
  sed -i 's/  - -e ..//' $ENV
fi

#Rename the conda environment if flag passed
if [ -z "$CONDA_ENV_NAME" ]; then
  CONDA_ENV_NAME=$(awk '/name/ {print $2}' $ENV)
else
  CONDA_OLD_NAME=$(awk '/name/ {print $2}' $ENV)
  sed -i "s/$CONDA_OLD_NAME/$CONDA_ENV_NAME/" $ENV
fi

#Create the conda environment and activate it
conda env create -f $ENV
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

#Finish building
if [ -n "$MPI_BUILD_PATH" ]; then
  pushd .
  cd ${MPI_BUILD_PATH}
  pwd
  MPICC=$(which mpicc)
  python setup.py build --mpicc $MPICC
  python setup.py install --user
  popd
  python -m pip -e ..
fi

rm $ENV

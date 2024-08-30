#!/bin/bash

# The example build script to build the source in Linux-like platform
set -e -u

cuda_arch=''
if [[ $@ == *"DOCOS_USE_CUDA=ON"* && $@ != *"DCMAKE_CUDA_ARCHITECTURES"* ]]; then
  nvidia=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits)
  cuda_arch=$(echo $nvidia | awk '{print $1}' | tr -d '.')
fi

OSNAME=$(uname -s)
if [ -z ${CPU_NUMBER+x} ]; then
  if [[ "$OSNAME" == "Darwin" ]]; then
    CPU_NUMBER=$(sysctl -n hw.logicalcpu)
  else
    CPU_NUMBER=$(nproc)
  fi
fi

BUILD_FLAVOR=RelWithDebInfo
target_dir=out/$OSNAME/$BUILD_FLAVOR
mkdir -p "$target_dir" && cd "$target_dir"

param="$@ ../../.."
if [ -n "$cuda_arch" ]; then
  param="$@ -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch ../../.."
fi
# it looks the parallel build on CI pipeline machine causes crashes.
cmake $param && cmake --build . --config $BUILD_FLAVOR  --parallel "${CPU_NUMBER}"

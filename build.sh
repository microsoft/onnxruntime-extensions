#!/bin/bash

# The example build script to build the source in Linux-like platform
set -e -x -u

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
# it looks the parallel build on CI pipeline machine causes crashes.
cmake "$@" ../../.. && cmake --build . --config $BUILD_FLAVOR  --parallel "${CPU_NUMBER}"

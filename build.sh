#!/bin/sh
# the script should be run with sh, bash on MacOS doesn't expand the alias
# The example build script to build the source in Linux-like platform
set -e -x -u

OSNAME=$(uname -s)
if [[ "$OSNAME" == "Darwin" ]]; then
  alias nproc="sysctl -n hw.logicalcpu"
fi

BUILD_FLAVOR=RelWithDebInfo
target_dir=out/$OSNAME/$BUILD_FLAVOR
mkdir -p "$target_dir" && cd "$target_dir"
if [ -z ${CPU_NUMBER+x} ]; then CPU_NUMBER=$(nproc); fi
# it looks the parallel build on CI pipeline machine causes crashes.
cmake "$@" ../../.. && cmake --build . --config $BUILD_FLAVOR  --parallel "${CPU_NUMBER}"

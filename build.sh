#!/bin/bash

# The example build script to build the source in Linux-like platform
set -e -x -u

OSNAME=$(uname -s)
BUILD_FLAVOR=RelWithDebInfo
target_dir=out/$OSNAME/$BUILD_FLAVOR
mkdir -p $target_dir && cd $target_dir

cmake "$@" ../../.. && cmake --build . --config $BUILD_FLAVOR

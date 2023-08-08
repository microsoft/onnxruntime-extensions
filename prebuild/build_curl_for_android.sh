#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -e
set -u
set -x

# export skip_checkout if you want to repeat a build
if [ -z ${skip_checkout+x} ]; then
    git clone https://github.com/leenjewel/openssl_for_ios_and_android.git
    cd openssl_for_ios_and_android
    git checkout ci-release-663da9e2
    # patch with fixes to build on linux with NDK 25 or later
    git apply ../build_curl_for_android_on_linux.patch
else
    echo "Skipping checkout and patch"
    cd openssl_for_ios_and_android
fi

cd tools

# we target Android API level 24
export api=24

# provide a specific architecture as an argument to the script to limit the build to that
# default is to build all
# valid architecture values: "arm" "arm64" "x86" "x86_64"
if [ $# -eq 1 ]; then
    arch=$1
    ./build-android-openssl.sh $arch
    ./build-android-nghttp2.sh $arch
    ./build-android-curl.sh $arch
else
    ./build-android-openssl.sh
    ./build-android-nghttp2.sh
    ./build-android-curl.sh
fi

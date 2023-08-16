#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

set -e
set -u

# change to the directory the script is in in case it's being executed from elsewhere
echo `pwd`
cd "$(dirname "$0")"

# to simplify we only fetch and patch if the directory doesn't exist. 
# if something fails during this stage you need to delete the directory to retry.
if [ ! -d "openssl_for_ios_and_android" ]; then
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

# we target Android API level 21 but allow override by environment variable
if [ -z ${ANDROID_API_LEVEL+x} ]; then 
    export api=21
else
    export api=${ANDROID_API_LEVEL}
fi

echo $api
# provide a specific architecture as an argument to the script to limit the build to that
# default is to build all
# valid architecture values: "arm" "arm64" "x86" "x86_64"
if [ $# -ge 1 ]; then
    arch=$1
    ./build-android-openssl.sh $arch
    # ./build-android-nghttp2.sh $arch
    ./build-android-curl.sh $arch
else
    ./build-android-openssl.sh
    # ./build-android-nghttp2.sh
    ./build-android-curl.sh
fi

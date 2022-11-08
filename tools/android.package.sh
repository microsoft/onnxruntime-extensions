#!/bin/bash
set -e -x -u

if [[ ! -f tools/android.package.sh ]]; then
    echo This tool has to run from the project root directory
    exit -1
fi

OSNAME=android
ABIS=('x86' 'arm64-v8a'  'x86_64' 'armeabi-v7a')
mkdir -p out/$OSNAME/Release/java/android/
for abi_name in "${ABIS[@]:1}";
do
    _BUILD_CFG="${abi_name} out/$OSNAME/Release_${abi_name}" ./build.android $@
    cp -R out/$OSNAME/Release_${abi_name}/java/android/${abi_name} out/$OSNAME/Release/java/android/
done

_BUILD_CFG="${ABIS[0]} out/$OSNAME/Release" ./build.android $@

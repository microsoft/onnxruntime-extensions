#!/bin/bash

BASE_DIR=onnxruntime_customops_integration

if [ ! -d $BASE_DIR ]
then	
    mkdir $BASE_DIR
fi
cd $BASE_DIR

git clone https://github.com/microsoft/onnxruntime.git
cd onnxruntime

# there is no stable version including webassembly
git checkout e6a3308db7c03a13e0f08b221b6770e17fc3a4ef
cd cmake/external
git clone git@github.com:microsoft/onnxruntime-extensions.git

cd ../..

cp cmake/external/onnxruntime-extensions/test/data/custom_op_negpos.onnx onnxruntime/test/testdata
cp cmake/external/onnxruntime-extensions/test/data/custom_op_string_lower.onnx onnxruntime/test/testdata
git apply cmake/external/onnxruntime-extensions/ci_build/onnxruntime_integration/onnxruntime_v1.8.patch

#get ready and begin building
cd ..
if [ ! -d build ]
then
    mkdir build
fi
python3 onnxruntime/tools/ci_build/build.py --build_dir build "$@"



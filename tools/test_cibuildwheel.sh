#!/bin/bash

if [[ "$OCOS_ENABLE_AZURE" == "1" ]]
then
    pushd $1/test
    python -m pip install onnxruntime
    python ./test_azure_ops.py
    popd
fi

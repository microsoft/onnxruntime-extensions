#!/bin/bash

if [[ "$OCOS_ENABLE_AZURE" == "1" ]]
then
    pushd $1/test
    python ./test_azure_ops.py
    popd
fi

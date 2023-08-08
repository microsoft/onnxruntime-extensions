#!/bin/bash

if [[ "$OCOS_ENABLE_AZURE" == "1" ]]
then
    pushd $1/test
    python -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    python -m pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly==1.16.0.dev20230806001
    python ./test_azure_ops.py
    popd
fi

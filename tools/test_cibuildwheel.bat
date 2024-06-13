@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    pushd %1\test
    python -m pip install coloredlogs flatbuffers numpy packaging protobuf sympy
    python -m pip install onnxruntime==1.18
    python test_azure_ops.py
    popd
)

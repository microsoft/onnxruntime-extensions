@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    pushd %1\test
    python -m pip install onnxruntime
    python test_azure_ops.py
    popd
)

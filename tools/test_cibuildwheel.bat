@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    pushd %1\test
    python install onnxruntime
    python test_azure_ops.py
    popd
)

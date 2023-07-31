@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    pushd %1\test
    python test_azure_ops.py
    popd
)

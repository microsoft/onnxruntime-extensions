@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    python test\test_azure_ops.bat
)
@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
    pushd %1\test
    python -m pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ ort-nightly
    python test_azure_ops.py
    popd
)

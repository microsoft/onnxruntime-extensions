@echo off
setlocal EnableDelayedExpansion

python -c "import onnxruntime_extensions as _ortx; import onnxruntime_extensions._extensions_pydll as _ext; print(_ext.__file__)"
if errorlevel 1 exit /b 1

for /f "delims=" %%i in ('python -c "import onnxruntime_extensions._extensions_pydll as m; print(m.__file__)"') do (
    set EXT_PATH=%%i
)

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3,12) else 1)"
if errorlevel 1 exit /b 0

python -m pip install -q abi3audit
if errorlevel 1 exit /b 1

abi3audit --assume-minimum-abi3 3.12 "%EXT_PATH%"
if errorlevel 1 exit /b 1

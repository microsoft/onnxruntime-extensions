@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
  pip install cmake
  if not exist "C:\Program Files\Miniconda3\python3.exe" (
    mklink "C:\Program Files\Miniconda3\python3.exe" "C:\Program Files\Miniconda3\python.exe"
  )
)
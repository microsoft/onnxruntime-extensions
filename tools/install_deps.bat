@echo off
if "%OCOS_ENABLE_AZURE%"=="1" (
  pip install cmake
  if not exist "%ProgramFiles%\Miniconda3\python3.exe" (
    mklink "%ProgramFiles%\Miniconda3\python3.exe" "%ProgramFiles%\Miniconda3\python.exe"
  )
)
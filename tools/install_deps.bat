@echo off
if "%1" == "install" (
  REM The vcpkg installation requires a cmake python module when being run from cibuildwheel
  pip install cmake

  REM This doesn't work on a CI as Miniconda isn't used. Given that, it's not clear what value it is providing.
  if not exist "%ProgramFiles%\Miniconda3\python3.exe" if exist "%ProgramFiles%\Miniconda3\python.exe" (
    mklink "%ProgramFiles%\Miniconda3\python3.exe" "%ProgramFiles%\Miniconda3\python.exe"
  )
)

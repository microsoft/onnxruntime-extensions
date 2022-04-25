@ECHO OFF
ECHO Copy this file to mybuild.bat and make any changes you deem necessary
SETLOCAL ENABLEDELAYEDEXPANSION
IF DEFINED VSINSTALLDIR GOTO :VSDEV_CMD
set VCVARS="NOT/EXISTED"
FOR %%I in (Enterprise Professional Community BuildTools^
  ) DO IF EXIST "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\%%I\VC\Auxiliary\Build\vcvars64.bat" (
       SET VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\%%I\VC\Auxiliary\Build\vcvars64.bat" )

IF NOT EXIST %VCVARS% GOTO :NOT_FOUND
ECHO Found %VCVARS%
CALL %VCVARS%

:VSDEV_CMD
set GENERATOR="Visual Studio 16 2019"
IF "%VisualStudioVersion:~0,2%" == "16" GOTO :START_BUILD
set GENERATOR="Visual Studio 17 2022"

:START_BUILD
mkdir .\out\Windows\ 2>NUL
cmake -G %GENERATOR% -A x64 %* -B out\Windows -S .
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
cmake --build out\Windows --config RelWithDebInfo
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
GOTO :EOF

:NOT_FOUND
ECHO "No Microsoft Visual Studio 2019 installation found!"
ECHO "  Or not run from Developer Command Prompt for VS 2022"
EXIT /B 1

ENDLOCAL

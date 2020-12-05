@ECHO OFF
ECHO Copy this file to mybuild.bat and make the changes as you needs
SETLOCAL ENABLEDELAYEDEXPANSION

set VCVARS="NOT/EXISTED"
FOR %%I in (Enterprise Professional Community BuildTools^
  ) DO IF EXIST "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\%%I\VC\Auxiliary\Build\vcvars64.bat" (
       SET VCVARS="%ProgramFiles(x86)%\Microsoft Visual Studio\2019\%%I\VC\Auxiliary\Build\vcvars64.bat" )

IF NOT EXIST %VCVARS% GOTO :NOT_FOUND
ECHO Found %VCVARS%
CALL %VCVARS%
mkdir .\out\Windows\ 2>NUL
cd out\Windows
cmake -G "Visual Studio 16 2019" -A x64 %* ..\..\
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
cmake --build . --config RelWithDebInfo
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
cd ..\..
GOTO :EOF

:NOT_FOUND
ECHO "No Microsoft Visual Studio 2019 installation found!"
EXIT /B 1

ENDLOCAL

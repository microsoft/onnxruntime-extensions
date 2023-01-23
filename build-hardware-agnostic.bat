@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
IF DEFINED VSINSTALLDIR GOTO :VSDEV_CMD
IF NOT DEFINED VSDEVCMD GOTO :NOT_FOUND

CALL "%VSDEVCMD%"

:VSDEV_CMD
set GENERATOR="Visual Studio 16 2019"
IF "%VisualStudioVersion:~0,2%" == "16" GOTO :START_BUILD
set GENERATOR="Visual Studio 17 2022"

:START_BUILD
set cmake_exe="%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
mkdir .\out\Windows\ 2>NUL
%cmake_exe% -G %GENERATOR% %* -B out\Windows -S .
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
%cmake_exe% --build out\Windows --config RelWithDebInfo
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
GOTO :EOF

:NOT_FOUND
ECHO "No Microsoft Visual Studio installation found!"
ECHO "  Please run build from Developer Command Prompt"
EXIT /B 1

ENDLOCAL

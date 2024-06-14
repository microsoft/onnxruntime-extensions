@ECHO OFF
SETLOCAL ENABLEDELAYEDEXPANSION
IF DEFINED VSINSTALLDIR GOTO :VSDEV_CMD
set _VSFINDER=%~dp0tools\get_vsdevcmd.ps1
for /f "tokens=* USEBACKQ" %%i in (
    `powershell -NoProfile -ExecutionPolicy Bypass -File "%_VSFINDER%"`) do call "%%i"

IF NOT DEFINED VSINSTALLDIR GOTO :NOT_FOUND

IF "%1" == "-A" GOTO :VSDEV_CMD
set GEN_PLATFORM=-A x64

:VSDEV_CMD
set GENERATOR="Visual Studio 16 2019"
IF "%VisualStudioVersion:~0,2%" == "16" GOTO :START_BUILD
set GENERATOR="Visual Studio 17 2022"

:START_BUILD
set cmake_exe="%VSINSTALLDIR%Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
mkdir .\out\Windows\ 2>NUL
%cmake_exe% -G %GENERATOR% %GEN_PLATFORM% %* -B out\Windows -S .
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
%cmake_exe% --build out\Windows --config RelWithDebInfo
IF %ERRORLEVEL% NEQ 0 EXIT /B %ERRORLEVEL%
GOTO :EOF

:NOT_FOUND
ECHO "No Microsoft Visual Studio installation found!"
ECHO "  Please run build from Developer Command Prompt"
EXIT /B 1

ENDLOCAL

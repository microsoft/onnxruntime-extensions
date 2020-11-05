@echo off
REM Copy this file to mybuild.bat and make the changes as you needs

REM call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake --version
call "%ProgramFiles(x86)%\Microsoft Visual Studio\2019\Enterprise\VC\Auxiliary\Build\vcvars64.bat"
mkdir .\out\Windows\ 2>NUL
cd out\Windows
cmake -G "Visual Studio 16 2019" -A x64 %* ..\..\
cmake --build . --config RelWithDebInfo 
cd ..\..

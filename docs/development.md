# Build and Development

This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.
## Python package
The package contains all custom operators and some Python scripts to manipulate the ONNX models.
- Install Visual Studio with C++ development tools on Windows, or gcc(>8.0) for Linux or xcode for macOS, and cmake on the unix-like platform. (**hints**: in Windows platform, if cmake bundled in Visual Studio was used, please specify the set _VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\<VERSION_YEAR>\<Edition>\Common7\Tools\VsDevCmd.bat_)
- If running on Windows, ensure that long file names are enabled, both for the [operating system](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd) and for git: `git config --system core.longpaths true`
- Prepare Python env and install the pip packages in the requirements.txt.
- `pip install .` to build and install the package.<br/> OR `pip install -e .` to install the package in the development mode, which is more friendly for the developer since the Python code change will take effect without having to copy the files to a different location in the disk.(**hints**: debug=1 in setup.cfg wil make C++ code be debuggable in a Python process.)

Test:
- 'pip install -r requirements-dev.txt' to install pip packages for development.
- run `pytest test` in the project root directory.

For a complete list of verified build configurations see [here](<./ci_matrix.md>)

## Java package
`bash ./build.sh -DOCOS_BUILD_JAVA=ON` to build jar package in out/<OS>/Release folder

## Android package
- pre-requisites: [Android Studio](https://developer.android.com/studio)

`bash ./tools/android.package.sh` to build the full AAR package or `bash ./build.android` to build a quick Android emulator package.

## iOS package
- TODO:

## Web-Assembly
ONNXRuntime-Extensions will be built as a static library and linked with ONNXRuntime due to the lack of dynamical library loading in WASM. Here are two additional arguments [–-use_extensions and --extensions_overridden_path](https://github.com/microsoft/onnxruntime/blob/860ba8820b72d13a61f0d08b915cd433b738ffdc/tools/ci_build/build.py#L416) on building onnxruntime to include ONNXRuntime-Extensions footprint in the ONNXRuntime package.

## The C++ share library
for any other cases, please run `build.bat` or `bash ./build.sh` to build the library. By default, the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. There is a unit test to help verify the build.

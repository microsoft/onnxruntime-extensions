# Build and Development

This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.

## Python package

The package contains all custom operators and some Python scripts to manipulate the ONNX models.

- Install Visual Studio with C++ development tools on Windows, or gcc(>8.0) for Linux or xcode for macOS, and cmake on the unix-like platform.
- If running on Windows, ensure that long file names are enabled, both for the [operating system](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd) and for git: `git config --system core.longpaths true`
- Make sure the Python development header/library files be installed, (like `apt-get install python3-dev` for Ubuntu Linux)
- `pip install .` to build and install the package.<br/> OR `pip install -e .` to install the package in the development mode, which is more friendly for the developer since the Python code change will take effect without having to copy the files to a different location in the disk.(**hints**: debug=1 in setup.cfg wil make C++ code be debuggable in a Python process.)
- Add the following argument `--config-settings "ortx-user-option=use-cuda"` in the pip command line to enable **CUDA** kernels for the package.
- The flags that can be used in --config-settings are:
  - use-cuda: enable CUDA kernel build in Python package.
  - no-azure: disable AzureOp kernel build in Python package.
  - no-opencv: disable operators based on OpenCV in build.
  - cc-debug: generate debug info for extensions binaries and disable C/C++ compiler optimization.
  - pp_api: enable pre-processing C ABI Python wrapper, `from onnxruntime_extensions.pp_api import *`
  - cuda-archs: specify the CUDA architectures(like 70, 85, etc.), and the multiple values can be combined with semicolon. The default value is nvidia-smi util output of GPU-0
  - ort\_pkg\_dir: specify ONNXRuntime package directory the extension project is depending on. This is helpful if you want to use some ONNXRuntime latest function which has not been involved in the official build

  For example:`pip install . --config-settings "ortx-user-option=use-cuda,cc-debug" `, This command builds CUDA kernels into the package and installs it, accompanied by the generation of debug information.

Test:

- 'pip install -r requirements-dev.txt' to install pip packages for development.
- run `pytest test` in the project root directory.

For a complete list of verified build configurations see [here](<./ci_matrix.md>)

## Java package

`bash ./build.sh -DOCOS_BUILD_JAVA=ON` to build jar package in out/<OS>/Release folder

## Android package

- pre-requisites: [Android Studio](https://developer.android.com/studio)

Use `./tools/android/build_aar.py` to build an Android AAR package.

## iOS package

Use `./tools/ios/build_xcframework.py` to build an iOS xcframework package.

## NuGet package

In order to build a local NuGet package for testing, run `nuget.exe pack ./nuget/WinOnlyNuget.nuspec` to build a NuGet package for Windows.

Note: you might need to update the src paths in the ./nuget/WinOnlyNuget.nuspec file if the appropriate ortextensions.dll files do not exist/are not in the given location.

## Web-Assembly

ONNXRuntime-Extensions will be built as a static library and linked with ONNXRuntime due to the lack of a good dynamic linking mechanism in WASM. Here are two additional arguments [–-use_extensions and --extensions_overridden_path](https://github.com/microsoft/onnxruntime/blob/860ba8820b72d13a61f0d08b915cd433b738ffdc/tools/ci_build/build.py#L416) on building onnxruntime to include ONNXRuntime-Extensions footprint in the ONNXRuntime package.

## The C++ shared library

For any alternative scenarios, execute the following commands:

- On Windows: Run `build.bat`.
- On Unix-based systems: Execute `bash ./build.sh`.

The generated DLL or library is typically located in the `out/<OS>/<FLAVOR>` directory. To validate the build, utilize the unit tests available in the `test/test_static_test` and `test/shared_test` directories.

**CUDA Build**  
The cuda build can be enabled with -DOCOS_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>

**VC Runtime static linkage**  
If you want to build the binary with VC Runtime static linkage, please add a parameter _-DCMAKE_MSVC_RUNTIME_LIBRARY="MultiThreaded$<$<CONFIG:Debug>:Debug>"_ when running build.bat

## Copyright guidance

Check this link [here](https://docs.opensource.microsoft.com/releasing/general-guidance/copyright-headers/) for source file copyright header.

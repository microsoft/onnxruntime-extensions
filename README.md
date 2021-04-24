# ONNXRuntime CustomOps
ONNXRuntime CustomOps Library is a comprehensive package to extent the ONNXRuntime with some capabilities via its custom ops API.
1. This repository provides a library of add-on custom operators for [ONNX Runtime](http://onnxruntime.ai). The package can be installed to run with ONNX Runtime for operators not natively supported by ORT. Learn more about [custom ops in ORT](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html). And the custom operator support list is in [docs/custom_text_ops.md](./docs/custom_text_ops.md)
2. Support PyOp feature to implement the custom op with a Python function.
3. Build all-in-one ONNX model from the pre/post processing code, go to [docs/pre_post_processing.md](docs/pre_post_processing.md) for details.
4. Support Python per operator debugging, checking ```hook_model_op``` in onnxruntime_customops Python package.

# Build and Development
This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.
## Python package
- Install Visual Studio with C++ development tools on Windows, or gcc for Linux or xcode for MacOS, and cmake on the unix-like platform. (**hints**: in Windows platform, if cmake bundled in Visual Studio was used, please specify the set _VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\<Edition>\VC\Auxiliary\Build\vcvars64.bat_)
- Prepare Python env and install the pip packages in the requirements.txt.
- `python setup.py install` to build and install the package.
- OR `python setup.py develop` to install the package in the development mode, which is more friendly for the developer since (re)installation is not needed with every build.

Test:
- run `pytest test` in the project root directory.

## The share library or DLL only
If only DLL/shared library is needed without any Python dependencies, please run `build.bat` or `bash ./build.sh` to build the library.
By default the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. There is a unit test to help verify the build.

# Contributing
This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Release
The package is currently release on test pypi
[onnxruntime-customops](https://test.pypi.org/project/onnxruntime-customops/).

# Changes

**0.0.2**: 

# License
[MIT License](LICENSE)

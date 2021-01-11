# ONNX Runtime Custom Ops Library
This repository provides a library of add-on custom operators for [ONNX Runtime](http://onnxruntime.ai). The package can be installed to run with ONNX Runtime for operators not natively supported by ORT. Learn more about [custom ops in ORT](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html). 

# Build and Development
This project supports Python and build it from source easily.
## Python package
- Install Visual Studio with C++ development tools on Windows, or gcc for Linux or xcode for MacOS, and cmake on the unix-like platform.
- Prepare Python env and install the pip packages in the requirements.txt.
- `python setup.py install` to build and install the package.
- OR `python setup.py develop` which is more friendly for you development.

Test:
- run `pytest test` in the project root directory.

## The share library or DLL only
If only DLL/shared library is needed without any Python bits, please run `build.bat` or `bash ./build.sh` to build the library.
By default the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. and there is a unit test to help verify the build.

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

# License
[MIT License](LICENSE)

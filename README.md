# ONNX Runtime Custom Ops Library
This repository provides a library of add-on custom operators for [ONNX Runtime](http://onnxruntime.ai). The package can be installed to run with ONNX Runtime for operators not natively supported by ORT. Learn more about [custom ops in ORT](https://www.onnxruntime.ai/docs/how-to/add-custom-op.html). 

# Getting started
Windows:
- Install Visual Studio with C++ development tools
- Copy build.bat to mybuild.bat and edit as needed. You may need to change "Enterprise" to "Community" depending on your Visual Studio version.
- Run mybuild.bat
- cd into `out/Windows/RelWithDebInfo` and run `pip install -e .`
- Run `python test/test_pyops.py` and run the repo root

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

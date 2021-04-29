# ONNXRuntime Extensions
[![Build Status](https://dev.azure.com/aiinfra/ONNX%20Converters/_apis/build/status/microsoft.ort-customops?repoName=microsoft%2Fonnxruntime-extensions&branchName=main)](https://dev.azure.com/aiinfra/ONNX%20Converters/_build/latest?definitionId=907&repoName=microsoft%2Fonnxruntime-extensions&branchName=main)

ONNXRuntime Extensions is a comprehensive package to extend the capability of the ONNX conversion and inference.
1. The CustomOp C++ library for [ONNX Runtime](http://onnxruntime.ai) on ONNXRuntime CustomOp API.
2. Support PyOp feature to implement the custom op with a Python function.
3. Build all-in-one ONNX model from the pre/post processing code, go to [docs/pre_post_processing.md](docs/pre_post_processing.md) for details.
4. Support Python per operator debugging, checking ```hook_model_op``` in onnxruntime_customops Python package.

# Quick Start
The following code shows how to run ONNX model and ONNXRuntime customop as a Python function.
```python
import numpy
from onnxruntime_customops import PyOrtFunction
# <ProjectDir>/tutorials/data/gpt2/gpt2_tok.onnx
encode = PyOrtFunction.from_model('gpt2_tok.onnx')
# https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
gpt2_core = PyOrtFunction.from_model('gpt2-lm-head-10.onnx')
# <ProjectDir>/tutorials/data/gpt2/gpt2_dec.onnx
decode = PyOrtFunction.from_model('gpt2_dec.onnx')

input_text = ['It is very cool to have']
output, *_ = gpt2_core(encode(input_text))
next_id = numpy.topk(output[:, -1, :], dim=-1)
print(' '.join(input_text[0], decode(next_id[0])))
```
This is a simplified version of GPT-2 inference for the demonstration only, The comprehensive solution on the GPT-2 model and its deviants are under development, and here is the [link](tutorials/gpt2_e2e.py) to the experimental.

## CustomOp conversion
The mainstream ONNX converters support the custom op generation if there is the operation from the original framework cannot be interpreted as ONNX standard operators. Check the following two examples on how to do this.
1. [CustomOp conversion by pytorch.onnx.exporter](tutorials/pytorch_custom_ops_tutorial.ipynb)
2. [CustomOp conversion by tf2onnx](tutorials/tf2onnx_custom_ops_tutorial.ipynb)

## Inference with CustomOp library
The CustomOp library was written with C++, so that it supports run the model in the native binaries. The following is the example of C++ version.
```C++
  // The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));

  // The regular ONNXRuntime invoking to run the model.
  Ort::Session session(env, model_uri, session_options);
  RunSession(session, inputs, outputs);
```
Of course, with Python language, the thing becomes much easier since PyOrtFunction will directly translate the ONNX model into a python function. But if the ONNXRuntime Custom Python API want to be used, the inference process will be
```python
import onnxruntime as _ort
from onnxruntime_customops import get_library_path as _lib_path

so = _ort.SessionOptions()
so.register_custom_ops_library(_lib_path())

# Run the ONNXRuntime Session.
# sess = _ort.InferenceSession(model, so)
# sess.run (...)
```

## More CustomOp
Welcome to contribute the customop C++ implementation directly in this repository, which will widely benefit other users. Besides C++, if you want to quickly verify the ONNX model with some custom operators with Python language, PyOp will help with that
```python
import numpy
from onnxruntime_customops import PyOp, onnx_op

# Implement the CustomOp by decorating a function with onnx_op
@onnx_op(op_type="Inverse", inputs=[PyOp.dt_float])
def inverse(x):
    # the user custom op implementation here:
    return numpy.linalg.inv(x)

# Run the model with this custom op
# model_func = PyOrtFunction(model_path)
# outputs = model_func(inputs)
# ...
```

# Build and Development
This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.
## Python package
- Install Visual Studio with C++ development tools on Windows, or gcc for Linux or xcode for MacOS, and cmake on the unix-like platform. (**hints**: in Windows platform, if cmake bundled in Visual Studio was used, please specify the set _VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\<Edition>\VC\Auxiliary\Build\vcvars64.bat_)
- Prepare Python env and install the pip packages in the requirements.txt.
- `python setup.py install` to build and install the package.
- OR `python setup.py develop` to install the package in the development mode, which is more friendly for the developer since (re)installation is not needed with every build.

Test:
- run `pytest test` in the project root directory.

## The share library for non-Python
If only DLL/shared library is needed without any Python dependencies, please run `build.bat` or `bash ./build.sh` to build the library.
By default the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. There is a unit test to help verify the build.

## The static library and link with ONNXRuntime
For sake of the binary size, the project can be built as a static library and link into ONNXRuntime. Here is [the script](ci_build/onnxruntime_integration/build_with_onnxruntime.sh) to this, which is especially usefully on building the mobile release.

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

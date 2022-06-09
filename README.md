# ONNX Runtime Extensions

[![Build Status](https://aiinfra.visualstudio.com/Lotus/_apis/build/status/onnxruntime-extensions/extensions.wheel?branchName=main)](https://aiinfra.visualstudio.com/Lotus/_build/latest?definitionId=1085&branchName=main)

## Introduction

ONNX Runtime Extensions is library that extends the capability of the ONNX conversion and inference with ONNX Runtime.

1. A library of common pre and post processing operators for vision, text, and nlp models for [ONNX Runtime](http://onnxruntime.ai) built using the ONNX Runtime CustomOp API.

2. A model augmentation API to integrate the pre and post processing steps into an ONNX model

3. The python operator feature that implements a custom operator with a Python function and can be used for testing and verification

4. A debugging tool called `hook_model_op`, which can be used for Python per operator debugging.

## Quick Start

### Installation

#### Install from PyPI

The latest release of onnxruntime-extensions is: 0.4.2.

```bash
pip install onnxruntime-extensions
```

#### Install from nightly builds


#### Install from source

1. Install the following pre-requisites

   * A C/C++ compiler for your operating system (gcc on Linux, Visual Studio on Windows, CLang on Mac)
   * [Cmake](https://cmake.org/)
   * [Git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)

2. If running on Windows, ensure that long file names are enabled, both for the [operating system](https://docs.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation?tabs=cmd) and for git: `git config --system core.longpaths true`

3. Install the package from source

   ```bash
   python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git
   ```

### Computer vision quick start

Build an augmented ONNX model with ImageNet pre and post processing.

```Python
import onnx
import torch
from onnxruntime_extensions import pnp

mnv2 = onnx.load_model('test/data/mobilev2.onnx')
augmented_model = pnp.SequentialProcessingModule(
    pnp.PreMobileNet(224),
    mnv2,
    pnp.PostMobileNet())

# the image size is dynamic, the 400x500 here is to get a fake input to enable export
fake_image_input = torch.ones(500, 400, 3).to(torch.uint8)
model_input_name = 'image'
pnp.export(augmented_model,
           fake_image_input,
           opset_version=11,
           output_path='mobilev2-aug.onnx',
           input_names=[model_input_name],
           dynamic_axes={model_input_name: [0, 1]})
```

The above python code will translate the ImageNet pre/post processing functions into an augmented model which can do inference on all platforms that ONNNXRuntime supports, like Android/iOS, without any Python runtime and the 3rd-party libraries dependency.

Note: On mobile platform, the ONNXRuntime package may not support all kernels required by the model, to ensure all the ONNX operator kernels were built into ONNXRuntime binaries, please use [ONNX Runtime Custom Build](https://onnxruntime.ai/docs/build/custom.html).


### Text pre and post processing

Build an augmented ONNX model with BERT pre and processing.

```python





### CustomOp Conversion

The mainstream ONNX converters support the custom op generation if the operation from the original framework cannot be interpreted as ONNX standard operators. Check the following two examples on how to do this.
1. [CustomOp conversion by pytorch.onnx.exporter](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/pytorch_custom_ops_tutorial.ipynb)
2. [CustomOp conversion by tf2onnx](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/tf2onnx_custom_ops_tutorial.ipynb)

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
from onnxruntime_extensions import get_library_path as _lib_path

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
from onnxruntime_extensions import PyOp, onnx_op

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
For sake of the binary size, the project can be built as a static library and link into ONNXRuntime. Here are two additional arguments [–-use_extensions and --extensions_overridden_path](https://github.com/microsoft/onnxruntime/blob/860ba8820b72d13a61f0d08b915cd433b738ffdc/tools/ci_build/build.py#L416) on building onnxruntime.

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

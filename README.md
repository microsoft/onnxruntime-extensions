# ONNXRuntime Extensions
[![Build Status](https://dev.azure.com/ms/onnxruntime-extensions/_apis/build/status/microsoft.ort-customops?branchName=main)](https://dev.azure.com/ms/onnxruntime-extensions/_build/latest?definitionId=512&branchName=main)

# Introduction
ONNXRuntime Extensions is a comprehensive package to extend the capability of the ONNX conversion and inference.
1. The CustomOp C++ library for [ONNX Runtime](http://onnxruntime.ai) on ONNXRuntime CustomOp API.
2. Integrate the pre/post processing steps into ONNX model which can be executed on all platforms that ONNXRuntime supported. check [pnp.export](onnxruntime_extensions/pnp/_unifier.py) for more details
3. Support PyOp feature to implement the custom op with a Python function.
4. Support Python per operator debugging, checking ```hook_model_op``` in onnxruntime_extensions Python package.

# Quick Start
### **Installation**
The package can be installed by standard pythonic way, ```pip install onnxruntime-extensions```.

To try the latest features in the source repo which haven't been released (cmake and the compiler like gcc required), the package can be installed as:
```python -m pip install git+https://github.com/microsoft/onnxruntime-extensions.git```

### **ImageNet Pre/Post Processing**
Build a full ONNX model with ImageNet pre/post processing
```Python
import onnx
import torch
from onnxruntime_extensions import pnp


mnv2 = onnx.load_model('test/data/mobilev2.onnx')
full_model = pnp.SequenceProcessingModule(
    pnp.PreMobileNet(224),
    mnv2,
    pnp.PostMobileNet())


# the image size is dynamic, the 400x500 here is to get a fake input to enable export
fake_image_input = torch.ones(500, 400, 3).to(torch.uint8)
full_model.forward(fake_image_input)
name_i = 'image'
pnp.export(full_model,
           fake_image_input,
           opset_version=11,
           output_path='temp_exmobilev2.onnx',
           input_names=[name_i],
           dynamic_axes={name_i: [0, 1]})
```
The above python code will translate the ImageNet pre/post processing functions into an all-in-one model which can do inference on all platforms that ONNNXRuntime supports, like Android/iOS, without any Python runtime and the 3rd-party libraries dependency.

Note: On mobile platform, the ONNXRuntime package may not support all kernels required by the model, to ensure all the ONNX operator kernels were built into ONNXRuntime binraries, please use [ONNX Runtime Mobile Custom Build](https://onnxruntime.ai/docs/tutorials/mobile/custom-build.html).

Here is a [tutorial](tutorials/imagenet_processing.ipynb) for pre/post processing details.

### **GPT-2 Pre/Post Processing**
The following code shows how to run ONNX model and ONNXRuntime customop more straightforwardly.
```python
import numpy
from onnxruntime_extensions import PyOrtFunction, VectorToString
# <ProjectDir>/tutorials/data/gpt-2/gpt2_tok.onnx
encode = PyOrtFunction.from_model('gpt2_tok.onnx')
# https://github.com/onnx/models/blob/master/text/machine_comprehension/gpt-2/model/gpt2-lm-head-10.onnx
gpt2_core = PyOrtFunction.from_model('gpt2-lm-head-10.onnx')
decode = PyOrtFunction.from_customop(VectorToString, map={' a': [257]}, unk='<unknown>')

input_text = ['It is very cool to have']
input_ids, *_ = encode(input_text)
output, *_ = gpt2_core(input_ids)
next_id = numpy.argmax(output[:, :, -1, :], axis=-1)
print(input_text[0] + decode(next_id).item())
```
This is a simplified version of GPT-2 inference for the demonstration only. The full solution of post-process can be checked [here](https://github.com/microsoft/onnxruntime/blob/ad9d2e2e891714e0911ccc3fa8b70f42025b4d56/docs/ContribOperators.md#commicrosoftbeamsearch)



## CustomOp Conversion
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
For sake of the binary size, the project can be built as a static library and link into ONNXRuntime. Here is [the script](https://github.com/microsoft/onnxruntime-extensions/blob/main/ci_build/onnxruntime_integration/build_with_onnxruntime.sh) to this, which is especially usefully on building the mobile release.

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

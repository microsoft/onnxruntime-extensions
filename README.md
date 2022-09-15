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

For a complete list of verified build configurations see [here](<./build_matrix.md>)

#### Install from PyPI

```bash
pip install onnxruntime-extensions
```

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

# Download the MobileNet V2 model from the ONNX model zoo
# https://github.com/onnx/models/blob/main/vision/classification/mobilenet/model/mobilenetv2-12.onnx

mnv2 = onnx.load_model('mobilenetv2-12.onnx')
augmented_model = pnp.SequentialProcessingModule(
    pnp.PreMobileNet(224),
    mnv2,
    pnp.PostMobileNet())

# The image size is dynamic, the 400x500 here is to get a fake input to enable export
fake_image_input = torch.ones(500, 400, 3).to(torch.uint8)
model_input_name = 'image'
pnp.export(augmented_model,
           fake_image_input,
           opset_version=11,
           output_path='mobilenetv2-aug.onnx',
           input_names=[model_input_name],
           dynamic_axes={model_input_name: [0, 1]})
```

The above python code will translate the ImageNet pre/post processing functions into an augmented model which can do inference on all platforms that ONNNXRuntime supports, like Android/iOS, without any Python runtime and the 3rd-party libraries dependency.

You can see a sample of the model augmentation code as well as a C# console app that runs the augmented model with ONNX Runtime [here](https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_sharp/image_classification)

Note: On mobile platform, the ONNXRuntime package may not support all kernels required by the model, to ensure all the ONNX operator kernels were built into ONNXRuntime binaries, please use [ONNX Runtime Custom Build](https://onnxruntime.ai/docs/build/custom.html).

### Text pre and post processing quick start

Obtain or export the base model.

```python
import torch

from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_path = "./" + model_name + ".onnx"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# set the model to inference mode
model.eval()

# Generate dummy inputs to the model. Adjust if neccessary
inputs = {
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long), # list of numerical ids for the tokenized text
        'attention_mask': torch.ones([1, 32], dtype=torch.long)      # dummy list of ones
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_llsen'}
torch.onnx.export(model,                                         # model being run
                  (inputs['input_ids'],
                   inputs['attention_mask']), 
                  model_path,                                    # where to save the model (can be a file or file-like object)
                  opset_version=11,                              # the ONNX version to export the model to
                  do_constant_folding=True,                      # whether to execute constant folding for optimization
                  input_names=['input_ids',
                               'input_mask'],                    # the model's input names
                  output_names=['output_logits'],                # the model's output names
                  dynamic_axes={'input_ids': symbolic_names,
                                'input_mask' : symbolic_names,
                                'output_logits' : symbolic_names}) # variable length axes
```

Build an augmented ONNX model with BERT pre and processing.

```python
from pathlib import Path
import torch
from transformers import AutoTokenizer
import onnx
from onnxruntime_extensions import pnp

# The fine-tuned HuggingFace model is exported to ONNX in the code snippet above
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_path = Path(model_name + ".onnx")

# mapping the BertTokenizer outputs into the onnx model inputs
def map_token_output(input_ids, attention_mask, token_type_ids):
    return input_ids.unsqueeze(0), token_type_ids.unsqueeze(0), attention_mask.unsqueeze(0)

# Post process the start and end logits
def post_process(*pred):
    output = torch.argmax(pred[0])
    return output

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_tokenizer = pnp.PreHuggingFaceBert(hf_tok=tokenizer)
bert_model = onnx.load_model(str(model_path))

augmented_model = pnp.SequentialProcessingModule(bert_tokenizer, map_token_output,
                                                 bert_model, post_process)

test_input = ["This is s test sentence"]

# create the final onnx model which includes pre- and post- processing.
augmented_model = pnp.export(augmented_model,
                             test_input,
                             opset_version=12,
                             input_names=['input'],
                             output_names=['output'],
                             output_path=model_name + '-aug.onnx',
                             dynamic_axes={'input': [0], 'output': [0]})
```

To run the augmented model with ONNX Runtime, you need to register the operators in the onnxruntime-extensions custom ops (including the BertTokenizer) library with ONNX Runtime.

```python
import onnxruntime
import onnxruntime_extensions

test_input = ["I don't really like tomatoes. They are too bitter"]

# Load the model
session_options = onnxruntime.SessionOptions()
session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
session = onnxruntime.InferenceSession('distilbert-base-uncased-finetuned-sst-2-english-aug.onnx', session_options)

# Run the model
results = session.run(["g2_output"], {"g1_it_2589433893008": test_input})

print(results[0])
```

The result is 0 when the sentiment is negative and 1 when the sentiment is positive.


## Register the custom operators in onnxruntime-extensions with ONNX Runtime

### C++

```c++
  // The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));

  // The regular ONNXRuntime invoking to run the model.
  Ort::Session session(env, model_uri, session_options);
  RunSession(session, inputs, outputs);
```

### Python

```python
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _lib_path

so = _ort.SessionOptions()
so.register_custom_ops_library(_lib_path())

# Run the ONNXRuntime Session.
# sess = _ort.InferenceSession(model, so)
# sess.run (...)
```

## Use exporters to generate graphs with custom operators

The PyTorch and TensorFlow converters support custom operator generation if the operation from the original framework cannot be interpreted as a standard ONNX operators. Check the following two examples on how to do this.

1. [CustomOp conversion by pytorch.onnx.exporter](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/pytorch_custom_ops_tutorial.ipynb)
2. [CustomOp conversion by tf2onnx](https://github.com/microsoft/onnxruntime-extensions/blob/main/tutorials/tf2onnx_custom_ops_tutorial.ipynb)


## Contribute a new operator to onnxruntime-extensions

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

## Build and Development

This project supports Python and can be built from source easily, or a simple cmake build without Python dependency.
### Python package

- Install Visual Studio with C++ development tools on Windows, or gcc for Linux or xcode for MacOS, and cmake on the unix-like platform. (**hints**: in Windows platform, if cmake bundled in Visual Studio was used, please specify the set _VCVARS=%ProgramFiles(x86)%\Microsoft Visual Studio\2019\<Edition>\VC\Auxiliary\Build\vcvars64.bat_)
- Prepare Python env and install the pip packages in the requirements.txt.
- `python setup.py install` to build and install the package.
- OR `python setup.py develop` to install the package in the development mode, which is more friendly for the developer since (re)installation is not needed with every build.

Test:
- run `pytest test` in the project root directory.

### The share library for non-Python

If only DLL/shared library is needed without any Python dependencies, please run `build.bat` or `bash ./build.sh` to build the library.
By default the DLL or the library will be generated in the directory `out/<OS>/<FLAVOR>`. There is a unit test to help verify the build.

### The static library and link with ONNXRuntime

For sake of the binary size, the project can be built as a static library and link into ONNXRuntime. Here are two additional arguments [–-use_extensions and --extensions_overridden_path](https://github.com/microsoft/onnxruntime/blob/860ba8820b72d13a61f0d08b915cd433b738ffdc/tools/ci_build/build.py#L416) on building onnxruntime.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## License

[MIT License](LICENSE)

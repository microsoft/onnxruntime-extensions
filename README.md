# ONNXRuntime-Extensions

[![Build Status](https://dev.azure.com/onnxruntime/onnxruntime/_apis/build/status%2Fonnxruntime-extensions.CI?branchName=main)](https://dev.azure.com/onnxruntime/onnxruntime/_build/latest?definitionId=213&branchName=main)

## What's ONNXRuntime-Extensions

Introduction: ONNXRuntime-Extensions is a C/C++ library that extends the capability of the ONNX models and inference with ONNX Runtime, via ONNX Runtime Custom Operator ABIs. It includes a set of [ONNX Runtime Custom Operator](https://onnxruntime.ai/docs/reference/operators/add-custom-op.html) to support the common pre- and post-processing operators for vision, text, and nlp models. And it supports multiple languages and platforms, like Python on Windows/Linux/macOS, some mobile platforms like Android and iOS, and Web-Assembly etc. The basic workflow is to enhance a ONNX model firstly and then do the model inference with ONNX Runtime and ONNXRuntime-Extensions package.


## Quickstart
The library can be utilized as either a C/C++ library or other advance language packages like Python, Java, C#, etc. To build it as a shared library, you can use the `build.bat` or `build.sh` scripts located in the root folder. The CMake build definition is available in the `CMakeLists.txt` file and can be modified by appending options to `build.bat` or `build.sh`, such as `build.bat -DOCOS_BUILD_SHARED_LIB=OFF`. For more details, please refer to the [C API documentation](./docs/c_api.md).

### **Python installation**
```bash
pip install onnxruntime-extensions
````
The nightly build is also available for the latest features, please refer to [nightly build](./docs/development.md#nightly-build)


## Usage

## 1. Generation of Pre-/Post-Processing ONNX Model
The `onnxruntime-extensions` Python package provides a convenient way to generate the ONNX processing graph. This can be achieved by converting the Huggingface transformer data processing classes into the desired format. For more detailed information, please refer to the API below:

```python
help(onnxruntime_extensions.gen_processing_models)
```
### NOTE:
The generation of model processing requires the **ONNX** package to be installed. The data processing models generated in this manner can be merged with other models using the [onnx.compose](https://onnx.ai/onnx/api/compose.html) if needed.

## 2. Using Extensions for ONNX Runtime inference

### Python
There are individual packages for the following languages, please install it for the build.
```python
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _lib_path

so = _ort.SessionOptions()
so.register_custom_ops_library(_lib_path())

# Run the ONNXRuntime Session, as ONNXRuntime docs suggested.
# sess = _ort.InferenceSession(model, so)
# sess.run (...)
```
### C++

```c++
  // The line loads the customop library into ONNXRuntime engine to load the ONNX model with the custom op
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options, custom_op_library_filename, &handle));

  // The regular ONNXRuntime invoking to run the model.
  Ort::Session session(env, model_uri, session_options);
  RunSession(session, inputs, outputs);
```
### Java
```java
var env = OrtEnvironment.getEnvironment();
var sess_opt = new OrtSession.SessionOptions();

/* Register the custom ops from onnxruntime-extensions */
sess_opt.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
```

### C#
```C#
SessionOptions options = new SessionOptions()
options.RegisterOrtExtensions()
session = new InferenceSession(model, options)
```


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

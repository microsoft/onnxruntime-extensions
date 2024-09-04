# ONNXRuntime Extensions C ABI

ONNXRuntime Extensions provides a C-style ABI for pre-processing. It offers support for tokenization, image processing, speech feature extraction, and more. You can compile the ONNXRuntime Extensions as either a static library or a dynamic library to access these APIs.

The C ABI header files are named `ortx_*.h` and can be found in the include folder. There are three types of data processing APIs available:

- [`ortx_tokenizer.h`](../include/ortx_tokenizer.h): Provides tokenization for LLM models.
- [`ortx_processor.h`](../include/ortx_processor.h): Offers image processing APIs for multimodels.
- [`ortx_extraction.h`](../include/ortx_extractor.h): Provides speech feature extraction for audio data processing to assist the Whisper model.

## ABI QuickStart

Most APIs accept raw data inputs such as audio, image compressed binary formats, or UTF-8 encoded text for tokenization.

**Tokenization:** You can create a tokenizer object using `OrtxCreateTokenizer` and then use the object to tokenize a text or decode the token ID into the text. A C-style code snippet is available [here](../test/pp_api_test/c_only_test.c).

**Image processing:** `OrtxCreateProcessor` can create an image processor object from a pre-defined workflow in JSON format to process image files into a tensor-like data type. An example code snippet can be found [here](../test/pp_api_test/test_processor.cc#L75).

**Audio feature extraction:** `OrtxCreateSpeechFeatureExtractor` creates a speech feature extractor to obtain log mel spectrum data as input for the Whisper model. An example code snippet can be found [here](../test/pp_api_test/test_feature_extraction.cc#L16).

**NB:** If onnxruntime-extensions is to build as a shared library, which requires the OCOS_ENABLE_AUDIO OCOS_ENABLE_CV2 OCOS_ENABLE_OPENCV_CODECS OCOS_ENABLE_GPT2_TOKENIZER build flags are ON to have a full function of binary. Only onnxruntime-extensions static library can be used for a minimal build with the selected operators, so in that case, the shared library build can be switched off by `-DOCOS_BUILD_SHARED_LIB=OFF`.

There is a simple Python wrapper on these C API in [pp_api](../onnxruntime_extensions/pp_api.py), which can have a easy access these APIs in Python code like

```Python
from onnxruntime_extensions.pp_api import Tokenizer
# the name can be the same one used by Huggingface transformers.AutoTokenizer
pp_tok = Tokenizer('google/gemma-2-2b') 
print(pp_tok.tokenize("what are you? \n 给 weiss ich, über was los ist \n"))
```

# ONNXRuntime Extensions C ABI

## Introduction

<span style="color:red">The C APIs in onnxruntime-extensions are experimental and subject to change.</span>

ONNXRuntime Extensions provides a C-style ABI for pre-processing. It offers support for tokenization, image processing, speech feature extraction, and more. You can compile the ONNXRuntime Extensions as either a static library or a dynamic library to access these APIs.

The C ABI header files are named `ortx_*.h` and can be found in the include folder. There are three types of data processing APIs available:

- [`ortx_tokenizer.h`](../include/ortx_tokenizer.h): Provides tokenization for LLM models.
- [`ortx_processor.h`](../include/ortx_processor.h): Offers image processing APIs for multimodel models.
- [`ortx_extraction.h`](../include/ortx_extractor.h): Provides speech feature extraction for audio data processing to assist the Whisper model.

## ABI QuickStart

Most APIs accept raw data inputs such as audio, image compressed binary formats, or UTF-8 encoded text for tokenization.

**Tokenization:** You can create a tokenizer object using `OrtxCreateTokenizer` and then use the object to tokenize a text or decode the token ID into the text. A C-style code snippet is available [here](../test/pp_api_test/test_tokenizer.cc#L448).

**Image processing:** `OrtxCreateProcessor` can create an image processor object from a pre-defined workflow in JSON format to process image files into a tensor-like data type. An example code snippet can be found [here](../test/pp_api_test/test_processor.cc#L16).

**Audio feature extraction:** `OrtxCreateSpeechFeatureExtractor` creates a speech feature extractor to obtain log mel spectrum data as input for the Whisper model. An example code snippet can be found [here](../test/pp_api_test/test_feature_extraction.cc#L15).

**Note:** To build onnxruntime-extensions as a shared library with full functionality, ensure the `OCOS_ENABLE_AUDIO` and `OCOS_ENABLE_GPT2_TOKENIZER` build flags are enabled. For a minimal build with selected operators, you can use the static library version and disable the shared library build by setting `-DOCOS_BUILD_SHARED_LIB=OFF`.

**Note:** A simple Python wrapper for these C APIs is available in [pp_api](../onnxruntime_extensions/pp_api.py), but it is not included in the default build. To enable it, use the extra build option `--config-settings "ortx-user-option=pp-api,no-opencv"`. For example, you can install it with the following command: `python3 -m pip install --config-settings "ortx-user-option=pp-api,no-opencv" git+https://github.com/microsoft/onnxruntime-extensions.git`. The following code demonstrates how to use the Python API to validate the tokenizer output.

```Python
from onnxruntime_extensions.pp_api import Tokenizer
# the name can be the same one used by Huggingface transformers.AutoTokenizer
pp_tok = Tokenizer('google/gemma-2-2b') 
print(pp_tok.tokenize("what are you? \n 给 weiss ich, über was los ist \n"))
```

## Hugging Face Tokenizer Data Compatibility

In the C API build, onnxruntime-extensions can directly load Hugging Face tokenizer data. Typically, a Hugging Face tokenizer includes `tokenizer.json` and `tokenizer_config.json` files, unless the model author has fully customized the tokenizer. The onnxruntime-extensions can seamlessly load these files. The following sections describe the supported fields in `tokenizer_config.json` and `tokenizer.json`.

1) The fields in `tokenizer_config.json` can affect the results of the onnxruntime-extensions tokenizer:

- `model_max_length`: the maximum length of the tokenized sequence.
- `bos_token`: the beginning of the sequence token, both `string` and `object` types are supported.
- `eos_token`: the end of the sequence token, both `string` and `object` types are supported.
- `unk_token`: the unknown token, both `string` and `object` types are supported.
- `pad_token`: the padding token, both `string` and `object` types are supported.
- `clean_up_tokenization_spaces`: whether to clean up the tokenization spaces.
- `tokenizer_class`: the tokenizer class.

2) The fields in `tokenizer.json` can affect the results of the onnxruntime-extensions tokenizer:

- `add_bos_token`: whether to add the beginning of the sequence token.
- `add_eos_token`: whether to add the end of the sequence token.
- `added_tokens`: the list of added tokens.
- `normalizer`: the normalizer, only 2 normalizers are supported, `Replace` and `precompiled_charsmap`.
- `pre_tokenizer`: Not directly used, but some properties can be inferred from other fields such as `decoders`.
- `post_processor`: `add_bos_token` and `add_eos_token` can be inferred from the `post_processor` field.
- `decoder/decoders`: the decoders, only `Replace` and `Strip` decoder steps are checked.
- `model/type`: the type of the model. If the type is missing, it will be treated as a Unigram model. Otherwise, the value of `model/type`.
- `model/vocab`: the vocabulary of the model.
- `model/merges`: the merges of the model.
- `model/end_of_word_suffix`: the end of the word suffix.
- `model/continuing_subword_prefix`: the continuing subword prefix.
- `model/byte_fallback`: Not supported.
- `model/unk_token_id`: the id of the unknown token.

3) `tokenizer_module.json` is a file that contains the user customized Python module information of the tokenizer, which is defined by onnxruntime-extensions, which is optional. The following fields are supported:

- `tiktoken_file`: the path of the tiktoken file base64 encoded vocab file, which can be loaded by `OrtxCreateTokenizer` too.
- `added_tokens`: same as `tokenizer.json`. If `tokenizer.json` does not contain `added_tokens` or the file does not exist, this field can be input by the user.

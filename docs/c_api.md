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

# ONNXRuntime Extensions C ABI

ONNXRuntime Extensions supports C style ABI to do pre-processing, it supports tokenizer, image-processing and speech  feature extraction and etc. You can compile the ONNXRuntime Extensions as a static library or dynamic library to access these APIs.

the C ABI header files are named as ortx_*.h in the include folder, and there are 3 types data processing API:
- ortx_tokenizer.h: the tokenization for LLM model
- ortx_processor.h: the image processing APIs for multimodel
- ortx_extraction.h: the speech feature extraction for audio data processing to help Whisper model

### ABI QuickStart

Most API accept the raw data input like audio, image compressed binary format, or UTF-8 encoded text for tokenization. 


# HuggingFace Compatibility

HuggingFace compatibility is a feature that allows you to use HuggingFace model data files with ONNXRuntime-Extensions for pre-/post-processing.


## HuggingFace Tokenizer

HuggingFace tokenizer always contains `tokenizer.json` and `tokenizer_config.json` files.
The following fields in `tokenizer_config.json` are supported in onnxruntime-extensions:

- `model_max_length`: the maximum length of the tokenized sequence.
- `bos_token`: the beginning of the sequence token, both `string` and `object` types are supported.
- `eos_token`: the end of the sequence token, both `string` and `object` types are supported.
- `unk_token`: the unknown token, both `string` and `object` types are supported.
- `pad_token`: the padding token, both `string` and `object` types are supported.
- `clean_up_tokenization_spaces`: whether to clean up the tokenization spaces.
- `tokenizer_class`: the tokenizer class.

The following fields in `tokenizer.json` are supported in onnxruntime-extensions:

- `add_bos_token`: whether to add the beginning of the sequence token.
- `add_eos_token`: whether to add the end of the sequence token.
- `added_tokens`: the list of added tokens.
- `normalizer`: the normalizer, only 2 normalizers are supported, `Replace` and `precompiled_charsmap`.
- `pre_tokenizer`: Not supported.
- `post_processor`: post process the tokenized sequence, only add bos/eos token in post processor is supported.
- `decoder/decoders`: the decoders, only `Replace` decoder step is supported.
- `model/type`: the type of the model, only `BPE` is supported.
- `model/vocab`: the vocabulary of the model.
- `model/merges`: the merges of the model.
- `model/end_of_word_suffix`: the end of the word suffix.
- `model/continuing_subword_prefix`: the continuing subword prefix.
- `model/byte_fallback`: Not supported.
- `model/unk_token_id`: the id of the unknown token.

`tokenizer_module.json` is a file that contains the user customized Python module information of the tokenizer, which is defined by onnxruntime-extensions, which is optional. The following fields are supported:

- `tiktoken_file`: the path of the tiktoken file base64 encoded vocab file.
- `added_tokens`: same as `tokenizer.json`. If `tokenizer.json` does not contain `added_tokens` or the file does not exist, this field can be input by the user.

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

`OrtxCreateTokenizerFromBlob` accepts tokenizer config and vocabulary bytes without a tokenizer directory. JSON vocabularies are classified from the supplied blob rather than a filename, including BPE vocabularies whose configured class normally selects Unigram. Config-only `added_tokens_decoder` metadata is retained just as in path-based loading; tokenizer-module added tokens keep the existing override precedence. Sidecar files are not discovered automatically: supply any tokenizer-module data explicitly in the blob.

**Image processing:** `OrtxCreateProcessor` can create an image processor object from a pre-defined workflow in JSON format to process image files into a tensor-like data type. An example code snippet can be found [here](../test/pp_api_test/test_processor.cc#L16).

**Audio feature extraction:** `OrtxCreateSpeechFeatureExtractor` creates a speech feature extractor to obtain log mel spectrum data as input for the Whisper model. An example code snippet can be found [here](../test/pp_api_test/test_feature_extraction.cc#L15).

### Bounded tokenization

Call `OrtxPrepareBoundedTokenization` during model initialization to prepare immutable token lookup data. Preparation supports byte-level BPE with complete byte-token coverage, native regex matchers, no fixed padding and no CLIP cleanup or SentencePiece transformation. Unsupported backends and regex fallback return `kOrtxErrorNotImplemented`; the existing unbounded tokenization APIs remain available and unchanged.

`OrtxTokenizeIntoBuffer` encodes one null-terminated, valid UTF-8 string into caller-owned token storage. Its `scratch_bytes` argument caps input-dependent temporary allocation requests through a per-call memory resource, without changing any process-global allocator. Model-owned lookup data, caller input/output buffers, fixed status bookkeeping and allocator overhead are outside that scratch allowance; it is not an RSS limit. The supported byte-level path needs at most input-byte-count plus two output slots, including optional BOS/EOS. It preserves added/special-token recognition and the tokenizer's `add_special_tokens` option.

Quota exhaustion, invalid input, missing preparation or insufficient output capacity leaves the output untouched and sets the returned count to zero. A failed call does not mutate tokenizer options or prevent a later call with a sufficient allowance. Calls on the same prepared tokenizer use independent scratch allowances.

### Finite video frames and optional file loading

`OrtxCreateRawFrames` snapshots an ordered clip of interleaved RGB8 frames and nondecreasing original-timeline timestamps. Validate every descriptor and aggregate byte quota before copying pixels; row padding is supported, but negative strides and short readable extents are not. Dispose the handle through `OrtxDispose`. `OrtxGetRawFrame` returns a borrowed view of the snapshot, not of the caller's mutable input.

`OrtxFramesPreProcess` resizes and normalizes real frames, then packs temporal groups through the configured `PatchImage` operation. Optional resolved geometry is per call and does not mutate the processor. This layer performs no sampling, model-token expansion, positional encoding or inference.

The `Resize` transform's `CUBIC_OPENCV` interpolation uses a non-antialiased four-tap cubic kernel (`a=-0.75`), half-pixel coordinates, replicated borders and floating coefficients. It clips and rounds to nearest-even RGB8 only after both axes. It requires no OpenCV runtime dependency. OpenCV builds using fixed-point coefficients can differ by one RGB level; bitwise equivalence across OpenCV backends is not promised. Existing `CUBIC` interpolation and its `coefficient_type` selection retain their PIL-style behavior.

`OrtxImagePreProcessWithOptions` accepts one `OrtxImagePreProcessOptions` per encoded image, each with `struct_size` equal to `sizeof(OrtxImagePreProcessOptions)`. It requires a `DecodeImage`/`ConvertRGB`/`Resize` prefix, checks the decoded source dimensions against the supplied plan, and applies the exact requested output dimensions without smart resizing or aspect-ratio selection. Geometry is validated before decoding, and the processor configuration is not mutated. The existing `OrtxImagePreProcess` remains the configured-geometry path.

`OrtxLoadVideo` produces the same handle from a local MP4/H.264 file. File loading is disabled by default; enable it with `OCOS_ENABLE_VIDEO_DECODER=ON`, the C API and vision processing, and `OCOS_FFMPEG_ROOT` pointing at a shared FFmpeg 8.x SDK. Disabled builds still export the loader and return `kOrtxErrorNotImplemented` with an unavailable message. Frame processing, images and package loading do not require FFmpeg in those builds.

Choose exactly one policy in `OrtxVideoLoadOptions`: positive FPS, positive count with an explicit finite range, or strictly increasing target times. Ranges are half-open `[start,end)`. Selection uses the first decoded presentation-order frame at or after each target, not average FPS or frame-index interpolation. FPS/count return distinct available frames and report collisions/underfill through `OrtxGetVideoMetadata` and `OrtxGetVideoSample`. Exact targets require distinct coverage and reject collisions/missing frames. A one-frame result is valid at this layer; model temporal minimums belong to the inference runtime.

Presentation seconds are elapsed from the original stream start, not from the selected range. Metadata preserves the original PTS origin and time base, and each sample records its selected raw PTS, original decode-order index and output-frame index. Uncovered count/FPS targets have output index `SIZE_MAX`. No timestamp sorting, guessed-rate fallback, model-dependent sampling or second sampling pass is performed. Decoding is sequential and bounded, while the demuxer's local-file seeks use the same read quota; there is no seek-based alternative selection policy.

The initial codec profile is progressive, square-pixel SDR H.264 with RGB8 output. Orthogonal track/frame rotation and reflection are applied once; unsupported projective transforms, HDR, interlacing and non-square pixels reject. Soundtracks are ignored. The demuxer cannot open external references or other files/protocols. Limits bound encoded size, cumulative reads, packets, decoded frames, generated targets, selected RGB bytes and dimensions. A cooperative 30-second deadline and a 16-megapixel codec allocation guard additionally bound work/geometry; the deadline is checked between decode steps and by FFmpeg's interrupt callback, not a preemptive cancellation of an operating-system file read. Decoder buffers and one-frame color/orientation staging are separate from the retained RGB quota; that quota is not a hard process-RSS limit. Corruption or quota exhaustion returns no partial handle. Inputs must remain regular, readable and unchanged during synchronous loading.

The local qualification SDK is FFmpeg 8.0.1, source SHA256 `05ee0b03119b45c0bdb4df654b96802e909e0a752f72e4fe3794f487229e5a41`, configured with `--disable-everything --disable-autodetect --disable-programs --disable-doc --disable-network --disable-x86asm --enable-shared --disable-static --enable-avcodec --enable-avformat --enable-swscale --enable-decoder=h264 --enable-parser=h264 --enable-demuxer=mov`. No GPL/nonfree components are enabled. The SDK is externally provisioned, not downloaded by ordinary builds. Codec-enabled deployment must supply the matching avformat, avcodec, avutil and swscale shared libraries through the platform loader, plus their applicable notices and corresponding-source delivery; follow the [FFmpeg distribution guidance](https://ffmpeg.org/legal.html). Enabled/disabled installed-artifact and Windows qualification remain separate delivery gates, not claims established by a linked Linux fixture run.

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
- `added_tokens_decoder`: token-ID metadata for added tokens, including tokens absent from `tokenizer.json`.

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

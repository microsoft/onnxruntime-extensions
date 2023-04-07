// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

#include <list>
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
#define DR_MP3_IMPLEMENTATION 1
#define DR_MP3_FLOAT_OUTPUT 1
#include "dr_mp3.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "string_utils.h"
#include "string_tensor.h"

struct KernelAudioDecoder : public BaseKernel {
 public:
  KernelAudioDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  enum {
    DEFAULT_STREAM = 0,
    WAV_STREAM,
    MP3_STREAM,
    FLAC_STREAM
  };

  int ReadStreamFormat(OrtKernelContext* context, const int64_t* p_data) {
    int stream_format = DEFAULT_STREAM;
    const OrtValue* ov_format = ort_.KernelContext_GetInput(context, 1);
    if (ov_format != nullptr) {
      std::vector<std::string> str_format;
      GetTensorMutableDataString(api_, ort_, context, ov_format, str_format);
      const char* formats[] = {"default",
                               "wav",
                               "mp3",
                               "flac"};
      for (auto fmt : formats) {
        if (str_format[0] == fmt) {
          break;
        }
        stream_format++;
      }
    }

    if (stream_format == DEFAULT_STREAM){
      auto p_stream = reinterpret_cast<char const*>(p_data);
      std::string_view marker(p_stream, 4);
      if (marker == "fLaC") {
        stream_format = FLAC_STREAM;
      } else if (marker == "RIFF") {
        stream_format = WAV_STREAM;
      } else if (marker[0] == char(-1) && marker[1] == char(-13)) {
        stream_format = MP3_STREAM;
      } else {
        ORTX_CXX_API_THROW("[AudioDecoder]: Unknown audio stream format", ORT_INVALID_ARGUMENT);
      }
    }
    return stream_format;
  }

  template<typename _CLS_AUDIO, typename _FN_DECODER>
  static size_t DrReadFrames(std::list<std::vector<float>>& frames, _FN_DECODER fx, _CLS_AUDIO& obj) {
    const size_t default_chunk_size = 4096;
    int64_t total_frames = 0;

    for (;;) {
      std::vector<float> buf;
      buf.resize(default_chunk_size * obj.channels);
      auto n_frames = fx(&obj, default_chunk_size, buf.data());
      if (n_frames <= 0) {
        break;
      }
      auto n_samples = n_frames * obj.channels;
      total_frames += n_samples;
      buf.resize(n_samples);
      frames.emplace_back(std::move(buf));
    }

    return total_frames;
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const int64_t* p_data = ort_.GetTensorData<int64_t>(input);

    OrtTensorDimensions input_dim(ort_, input);
    if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
      ORTX_CXX_API_THROW("[AudioDecoder]: Expect input dimension [n] or [1,n].", ORT_INVALID_ARGUMENT);
    }

    auto stream_format = ReadStreamFormat(context, p_data);

    int64_t total_frames = 0;
    std::list<std::vector<float>> lst_frames;

    if (stream_format == MP3_STREAM) {
      drmp3 mp3_obj;
      if (!drmp3_init_memory(&mp3_obj, p_data, input_dim.Size(), nullptr)) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on MP3 stream.", ORT_RUNTIME_EXCEPTION);
      }
      total_frames = DrReadFrames(lst_frames, drmp3_read_pcm_frames_f32, mp3_obj);

    } else if (stream_format == FLAC_STREAM) {
      drflac* flac_obj = drflac_open_memory(p_data, input_dim.Size(), nullptr);
      if (flac_obj == nullptr) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on FLAC stream.", ORT_RUNTIME_EXCEPTION);
      }
      auto flac_obj_closer = gsl::finally([flac_obj]() { drflac_close(flac_obj); });
      total_frames = DrReadFrames(lst_frames, drflac_read_pcm_frames_f32, *flac_obj);

    } else {
      drwav wav_obj;
      if (!drwav_init_memory(&wav_obj, p_data, input_dim.Size(), nullptr)) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on Wav stream.", ORT_RUNTIME_EXCEPTION);
      }
      total_frames = DrReadFrames(lst_frames, drwav_read_pcm_frames_f32, wav_obj);
    }

    std::vector<int64_t> dim_out = {1, total_frames};
    OrtValue* v = ort_.KernelContext_GetOutput(context, 0, dim_out.data(), dim_out.size());
    float* p_output = ort_.GetTensorMutableData<float>(v);
    int64_t offset = 0;
    for (auto& _b : lst_frames) {
      std::copy(_b.begin(), _b.end(), p_output + offset);
      offset += _b.size();
    }
  }
};

struct CustomOpAudioDecoder : OrtW::CustomOpBase<CustomOpAudioDecoder, KernelAudioDecoder> {
  const char* GetName() const {
    return "AudioDecoder";
  }

  size_t GetInputTypeCount() const {
    return 2;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8 : ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }

  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t index) const {
    return index == 0 ? OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED
      : OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_OPTIONAL;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  }
};

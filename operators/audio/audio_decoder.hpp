// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

#include <list>
#include <map>
#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
#define DR_MP3_IMPLEMENTATION 1
#define DR_MP3_FLOAT_OUTPUT 1
#include "dr_mp3.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include <gsl/util>
#include "string_utils.h"
#include "string_tensor.h"

struct KernelAudioDecoder : public BaseKernel {
 public:
  KernelAudioDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  enum class AudioStreamType {
    kDefault = 0,
    kWAV,
    kMP3,
    kFLAC
  };

  AudioStreamType ReadStreamFormat(OrtKernelContext* context, const uint8_t* p_data) {
    static const std::map<std::string, AudioStreamType> format_mapping = {
        {"default", AudioStreamType::kDefault},
        {"wav", AudioStreamType::kWAV},
        {"mp3", AudioStreamType::kMP3},
        {"flac", AudioStreamType::kFLAC}};

    AudioStreamType stream_format = AudioStreamType::kDefault;
    const OrtValue* ov_format = ort_.KernelContext_GetInput(context, 1);
    if (ov_format != nullptr) {
      std::vector<std::string> str_format;
      GetTensorMutableDataString(api_, ort_, context, ov_format, str_format);
      auto pos = format_mapping.find(str_format[0]);
      if (pos == format_mapping.end()) {
        ORTX_CXX_API_THROW(MakeString(
                               "[AudioDecoder]: Unknown audio stream format: ", str_format),
                           ORT_INVALID_ARGUMENT); }
      stream_format = pos->second;
    }

    if (stream_format == AudioStreamType::kDefault) {
      auto p_stream = reinterpret_cast<char const*>(p_data);
      std::string_view marker(p_stream, 4);
      if (marker == "fLaC") {
        stream_format = AudioStreamType::kFLAC;
      } else if (marker == "RIFF") {
        stream_format = AudioStreamType::kWAV;
      } else if (marker[0] == char(0xFF) && (marker[1] | 0x1F) == char(0xFF)) {
        // http://www.mp3-tech.org/programmer/frame_header.html
        // only detect the 8 + 3 bits sync word
        stream_format = AudioStreamType::kMP3;
      } else {
        ORTX_CXX_API_THROW("[AudioDecoder]: Cannot detect audio stream format", ORT_INVALID_ARGUMENT);
      }
    }

    return stream_format;
  }

  template <typename TY_AUDIO, typename FX_DECODER>
  static size_t DrReadFrames(std::list<std::vector<float>>& frames, FX_DECODER fx, TY_AUDIO& obj) {
    const size_t default_chunk_size = 1024 * 256;
    int64_t total_buf_size = 0;

    for (;;) {
      std::vector<float> buf;
      buf.resize(default_chunk_size * obj.channels);
      auto n_frames = fx(&obj, default_chunk_size, buf.data());
      if (n_frames <= 0) {
        break;
      }
      auto data_size = n_frames * obj.channels;
      total_buf_size += data_size;
      buf.resize(data_size);
      frames.emplace_back(std::move(buf));
    }

    return total_buf_size;
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
    const uint8_t* p_data = ort_.GetTensorData<uint8_t>(input);

    OrtTensorDimensions input_dim(ort_, input);
    if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
      ORTX_CXX_API_THROW("[AudioDecoder]: Expect input dimension [n] or [1,n].", ORT_INVALID_ARGUMENT);
    }

    auto stream_format = ReadStreamFormat(context, p_data);

    int64_t total_buf_size = 0;
    std::list<std::vector<float>> lst_frames;

    if (stream_format == AudioStreamType::kMP3) {
      auto mp3_obj_ptr = std::make_unique<drmp3>();
      if (!drmp3_init_memory(mp3_obj_ptr.get(), p_data, input_dim.Size(), nullptr)) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on MP3 stream.", ORT_RUNTIME_EXCEPTION);
      }
      total_buf_size = DrReadFrames(lst_frames, drmp3_read_pcm_frames_f32, *mp3_obj_ptr);

    } else if (stream_format == AudioStreamType::kFLAC) {
      drflac* flac_obj = drflac_open_memory(p_data, input_dim.Size(), nullptr);
      auto flac_obj_closer = gsl::finally([flac_obj]() { drflac_close(flac_obj); });
      if (flac_obj == nullptr) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on FLAC stream.", ORT_RUNTIME_EXCEPTION);
      }
      total_buf_size = DrReadFrames(lst_frames, drflac_read_pcm_frames_f32, *flac_obj);

    } else {
      drwav wav_obj;
      if (!drwav_init_memory(&wav_obj, p_data, input_dim.Size(), nullptr)) {
        ORTX_CXX_API_THROW("[AudioDecoder]: unexpected error on WAV stream.", ORT_RUNTIME_EXCEPTION);
      }
      total_buf_size = DrReadFrames(lst_frames, drwav_read_pcm_frames_f32, wav_obj);
    }

    std::vector<int64_t> dim_out = {1, total_buf_size};
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

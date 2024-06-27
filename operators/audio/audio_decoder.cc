// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <map>
#include <memory>
#include <gsl/util>

#include "audio_decoder.h"

#define DR_FLAC_IMPLEMENTATION
#include "dr_flac.h"
#define DR_MP3_IMPLEMENTATION 1
#define DR_MP3_FLOAT_OUTPUT 1
#include "dr_mp3.h"
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#include "narrow.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "sampling.h"

OrtStatusPtr AudioDecoder::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  auto status = OrtW::GetOpAttribute(info, "downsampling_rate", downsample_rate_);
  if (!status) {
    status = OrtW::GetOpAttribute(info, "stereo_to_mono", stereo_mixer_);
  }

  return status;
}

AudioDecoder::AudioStreamType AudioDecoder::ReadStreamFormat(const uint8_t* p_data, const std::string& str_format,
                                                             OrtxStatus& status) const {
  const std::map<std::string, AudioStreamType> format_mapping = {{"default", AudioStreamType::kDefault},
                                                                 {"wav", AudioStreamType::kWAV},
                                                                 {"mp3", AudioStreamType::kMP3},
                                                                 {"flac", AudioStreamType::kFLAC}};

  AudioStreamType stream_format = AudioStreamType::kDefault;
  if (str_format.length() > 0) {
    auto pos = format_mapping.find(str_format);
    if (pos == format_mapping.end()) {
      status = {kOrtxErrorInvalidArgument,
                MakeString("[AudioDecoder]: Unknown audio stream format: ", str_format).c_str()};
      return stream_format;
    }
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
      status = {kOrtxErrorInvalidArgument, "[AudioDecoder]: Cannot detect audio stream format"};
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

OrtxStatus AudioDecoder::Compute(const ortc::Tensor<uint8_t>& input, const std::optional<std::string> format,
                                 ortc::Tensor<float>& output0) const {
  const uint8_t* p_data = input.Data();
  auto input_dim = input.Shape();
  OrtxStatus status;
  if (!((input_dim.size() == 1) || (input_dim.size() == 2 && input_dim[0] == 1))) {
    return {kOrtxErrorInvalidArgument, "[AudioDecoder]: Expect input dimension [n] or [1,n]."};
  }

  std::string str_format;
  if (format) {
    str_format = *format;
  }
  auto stream_format = ReadStreamFormat(p_data, str_format, status);
  if (status) {
    return status;
  }

  int64_t total_buf_size = 0;
  std::list<std::vector<float>> lst_frames;
  int64_t orig_sample_rate = 0;
  int64_t orig_channels = 0;

  if (stream_format == AudioStreamType::kMP3) {
    auto mp3_obj_ptr = std::make_unique<drmp3>();
    if (!drmp3_init_memory(mp3_obj_ptr.get(), p_data, input.NumberOfElement(), nullptr)) {
      status = {kOrtxErrorCorruptData, "[AudioDecoder]: unexpected error on MP3 stream."};
      return status;
    }
    orig_sample_rate = mp3_obj_ptr->sampleRate;
    orig_channels = mp3_obj_ptr->channels;
    total_buf_size = DrReadFrames(lst_frames, drmp3_read_pcm_frames_f32, *mp3_obj_ptr);

  } else if (stream_format == AudioStreamType::kFLAC) {
    drflac* flac_obj = drflac_open_memory(p_data, input.NumberOfElement(), nullptr);
    auto flac_obj_closer = gsl::finally([flac_obj]() { drflac_close(flac_obj); });
    if (flac_obj == nullptr) {
      status = {kOrtxErrorCorruptData, "[AudioDecoder]: unexpected error on FLAC stream."};
      return status;
    }
    orig_sample_rate = flac_obj->sampleRate;
    orig_channels = flac_obj->channels;
    total_buf_size = DrReadFrames(lst_frames, drflac_read_pcm_frames_f32, *flac_obj);

  } else {
    drwav wav_obj;
    if (!drwav_init_memory(&wav_obj, p_data, input.NumberOfElement(), nullptr)) {
      status = {kOrtxErrorCorruptData, "[AudioDecoder]: unexpected error on WAV stream."};
      return status;
    }
    orig_sample_rate = wav_obj.sampleRate;
    orig_channels = wav_obj.channels;
    total_buf_size = DrReadFrames(lst_frames, drwav_read_pcm_frames_f32, wav_obj);
  }

  if (downsample_rate_ != 0 && orig_sample_rate < downsample_rate_) {
    status = {kOrtxErrorCorruptData, "[AudioDecoder]: only down-sampling supported."};
    return status;
  }

  // join all frames
  std::vector<float> buf;
  buf.resize(total_buf_size);
  int64_t offset = 0;
  for (auto& _b : lst_frames) {
    std::copy(_b.begin(), _b.end(), buf.begin() + offset);
    offset += _b.size();
  }

  // mix the stereo channels into mono channel
  if (stereo_mixer_ && orig_channels > 1) {
    if (buf.size() > 1) {
      for (size_t i = 0; i < buf.size() / 2; ++i) {
        buf[i] = (buf[i * 2] + buf[i * 2 + 1]) / 2;
      }
      buf.resize(buf.size() / 2);
    }
  }

  if (downsample_rate_ != 0 && downsample_rate_ != orig_sample_rate) {
    // A lowpass filter on buf audio data to remove high frequency noise
    ButterworthLowpass filter(0.5 * downsample_rate_, 1.0 * orig_sample_rate);
    std::vector<float> filtered_buf = filter.Process(buf);
    // downsample the audio data
    KaiserWindowInterpolation::Process(filtered_buf, buf, 1.0f * orig_sample_rate, 1.0f * downsample_rate_);
  }

  std::vector<int64_t> dim_out = {1, ort_extensions::narrow<int64_t>(buf.size())};
  float* p_output = output0.Allocate(dim_out);
  std::copy(buf.begin(), buf.end(), p_output);
  return status;
}

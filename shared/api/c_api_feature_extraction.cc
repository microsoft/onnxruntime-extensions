// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speech_extractor.h"

#include "c_api_utils.hpp"
#include <math/energy_stft_segmentation.hpp>
#include "audio/audio_decoder.h"

using namespace ort_extensions;

class RawAudiosObject : public OrtxObjectImpl {
 public:
  RawAudiosObject() : OrtxObjectImpl(extObjectKind_t::kOrtxKindRawAudios) {}
  ~RawAudiosObject() override = default;

  std::unique_ptr<AudioRawData[]> audios_;
  size_t num_audios_{};
};

extError_t ORTX_API_CALL OrtxCreateRawAudios(OrtxRawAudios** audios, const void* data[], const int64_t sizes[],
                                             size_t num_audios) {
  if (audios == nullptr || data == nullptr || sizes == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto audios_obj = std::make_unique<RawAudiosObject>();
  audios_obj->audios_ = std::make_unique<AudioRawData[]>(num_audios);
  audios_obj->num_audios_ = num_audios;
  for (size_t i = 0; i < num_audios; ++i) {
    audios_obj->audios_[i].resize(sizes[i]);
    std::copy_n(static_cast<const std::byte*>(data[i]), sizes[i], audios_obj->audios_[i].data());
  }

  *audios = static_cast<OrtxRawAudios*>(audios_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxLoadAudios(OrtxRawAudios** raw_audios, const char* const* audio_paths, size_t num_audios) {
  if (raw_audios == nullptr || audio_paths == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto audios_obj = std::make_unique<RawAudiosObject>();
  auto [audios, num] =
      ort_extensions::LoadRawData<char const* const*, AudioRawData>(audio_paths, audio_paths + num_audios);
  if (num == 0) {
    ReturnableStatus::last_error_message_ = "No audio data loaded";
    return kOrtxErrorInvalidArgument;
  }
  audios_obj->audios_ = std::move(audios);
  audios_obj->num_audios_ = num;

  *raw_audios = static_cast<OrtxRawAudios*>(audios_obj.release());
  return extError_t();
}

extError_t ORTX_API_CALL OrtxCreateSpeechFeatureExtractor(OrtxFeatureExtractor** extractor, const char* def) {
  if (extractor == nullptr || def == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto extractor_ptr = std::make_unique<SpeechFeatureExtractor>();
  ReturnableStatus status = extractor_ptr->Init(def);
  if (status.IsOk()) {
    *extractor = static_cast<OrtxFeatureExtractor*>(extractor_ptr.release());
  } else {
    *extractor = nullptr;
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxSpeechLogMel(OrtxFeatureExtractor* extractor, OrtxRawAudios* raw_audios,
                                          OrtxTensorResult** result) {
  if (extractor == nullptr || raw_audios == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto extractor_ptr = static_cast<SpeechFeatureExtractor*>(extractor);
  auto audios_obj = static_cast<RawAudiosObject*>(raw_audios);

  auto ts_result = std::make_unique<TensorResult>();
  std::unique_ptr<ortc::Tensor<float>> log_mel[1];
  ReturnableStatus status =
      extractor_ptr->DoCall(ort_extensions::span(audios_obj->audios_.get(), audios_obj->num_audios_), log_mel[0]);
  if (status.IsOk()) {
    std::vector<std::unique_ptr<ortc::TensorBase>> tensors;
    std::transform(log_mel, log_mel + 1, std::back_inserter(tensors),
                   [](auto& ts) { return std::unique_ptr<ortc::TensorBase>(ts.release()); });
    ts_result->SetTensors(std::move(tensors));
    *result = ts_result.release();
  } else {
    *result = nullptr;
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxSplitSignalSegments(const OrtxTensor* input, const OrtxTensor* sr_tensor,
                                                 const OrtxTensor* frame_ms_tensor, const OrtxTensor* hop_ms_tensor,
                                                 const OrtxTensor* energy_threshold_db_tensor, OrtxTensor* output0) {
  if (input == nullptr || sr_tensor == nullptr || frame_ms_tensor == nullptr || hop_ms_tensor == nullptr ||
      energy_threshold_db_tensor == nullptr || output0 == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }
  const ortc::Tensor<float>& input_tensor = *reinterpret_cast<const ortc::Tensor<float>*>(input);
  const ortc::Tensor<int64_t>& sr_t = *reinterpret_cast<const ortc::Tensor<int64_t>*>(sr_tensor);
  const ortc::Tensor<int64_t>& frame_t = *reinterpret_cast<const ortc::Tensor<int64_t>*>(frame_ms_tensor);
  const ortc::Tensor<int64_t>& hop_t = *reinterpret_cast<const ortc::Tensor<int64_t>*>(hop_ms_tensor);
  const ortc::Tensor<float>& threshold_t = *reinterpret_cast<const ortc::Tensor<float>*>(energy_threshold_db_tensor);
  ortc::Tensor<int64_t>& output_t = *reinterpret_cast<ortc::Tensor<int64_t>*>(output0);

  OrtStatusPtr status = split_signal_segments(input_tensor, sr_t, frame_t, hop_t, threshold_t, output_t);
  if (status) {
    ReturnableStatus::last_error_message_ = "split_signal_segments failed";
    return kOrtxErrorInvalidArgument;
  }

  return extError_t();
}

extError_t ORTX_API_CALL OrtxMergeSignalSegments(const OrtxTensor* segments_tensor,
                                                 const OrtxTensor* merge_gap_ms_tensor, OrtxTensor* output0) {
  if (segments_tensor == nullptr || merge_gap_ms_tensor == nullptr || output0 == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const ortc::Tensor<int64_t>& seg_t = *reinterpret_cast<const ortc::Tensor<int64_t>*>(segments_tensor);
  const ortc::Tensor<int64_t>& gap_t = *reinterpret_cast<const ortc::Tensor<int64_t>*>(merge_gap_ms_tensor);
  ortc::Tensor<int64_t>& output_t = *reinterpret_cast<ortc::Tensor<int64_t>*>(output0);

  OrtStatusPtr status = merge_signal_segments(seg_t, gap_t, output_t);
  if (status) {
    ReturnableStatus::last_error_message_ = "merge_signal_segments failed";
    return kOrtxErrorInvalidArgument;
  }

  return kOrtxOK;
}

extError_t ORTX_API_CALL OrtxFeatureExtraction(OrtxFeatureExtractor* extractor, OrtxRawAudios* raw_audios,
                                               OrtxTensorResult** result) {
  if (extractor == nullptr || raw_audios == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto extractor_ptr = static_cast<SpeechFeatureExtractor*>(extractor);
  auto audios_obj = static_cast<RawAudiosObject*>(raw_audios);

  auto result_ptr = std::make_unique<TensorResult>();
  ReturnableStatus status =
      extractor_ptr->Preprocess(ort_extensions::span(audios_obj->audios_.get(), audios_obj->num_audios_), *result_ptr);
  if (status.IsOk()) {
    *result = static_cast<OrtxTensorResult*>(result_ptr.release());
  } else {
    *result = nullptr;
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxDecodeAudio(OrtxRawAudios* raw_audios, size_t index, int64_t target_sample_rate,
                                         OrtxTensorResult** result) {
  if (raw_audios == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto* audios_obj = static_cast<RawAudiosObject*>(raw_audios);
  if (index >= audios_obj->num_audios_) {
    ReturnableStatus::last_error_message_ = "Audio index out of range";
    return kOrtxErrorInvalidArgument;
  }

  // Configure decoder
  AudioDecoder decoder;
  std::unordered_map<std::string, std::variant<std::int64_t, std::vector<std::int64_t>>> attrs;
  if (target_sample_rate > 0) {
    attrs["target_sample_rate"] = target_sample_rate;
  }
  attrs["stereo_to_mono"] = int64_t{1};
  attrs["max_samples"] = int64_t{0};  // No truncation
  ReturnableStatus status = decoder.Init(attrs);
  if (!status.IsOk()) {
    *result = nullptr;
    return status.Code();
  }

  // Wrap raw audio bytes as a uint8 tensor for the decoder
  auto& audio_bytes = audios_obj->audios_[index];
  std::vector<int64_t> input_shape = {1, static_cast<int64_t>(audio_bytes.size())};
  ortc::Tensor<uint8_t> input_tensor(&CppAllocator::Instance());
  auto* input_data = input_tensor.Allocate(input_shape);
  std::memcpy(input_data, audio_bytes.data(), audio_bytes.size());

  // Decode to float32 PCM
  auto pcm_tensor = std::make_unique<ortc::Tensor<float>>(&CppAllocator::Instance());
  auto sr_tensor = std::make_unique<ortc::Tensor<int64_t>>(&CppAllocator::Instance());
  status = decoder.ComputeNoOpt2(input_tensor, *pcm_tensor, *sr_tensor);
  if (!status.IsOk()) {
    *result = nullptr;
    return status.Code();
  }

  // Pack results: [0] = pcm float32, [1] = sample rate int64
  auto ts_result = std::make_unique<TensorResult>();
  std::vector<std::unique_ptr<ortc::TensorBase>> tensors;
  tensors.push_back(std::move(pcm_tensor));
  tensors.push_back(std::move(sr_tensor));
  ts_result->SetTensors(std::move(tensors));
  *result = static_cast<OrtxTensorResult*>(ts_result.release());

  return extError_t();
}

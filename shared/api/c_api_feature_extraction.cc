// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speech_extractor.h"

#include "c_api_utils.hpp"

using namespace ort_extensions;

class RawAudiosObject : public OrtxObjectImpl {
 public:
  RawAudiosObject() : OrtxObjectImpl(extObjectKind_t::kOrtxKindRawAudios) {}
  ~RawAudiosObject() override = default;

  std::unique_ptr<AudioRawData[]> audios_;
  size_t num_audios_{};
};

extError_t ORTX_API_CALL
OrtxCreateRawAudios(OrtxRawAudios** audios, const void* data[], const int64_t sizes[],  size_t num_audios) {
  if (audios == nullptr || data == nullptr || sizes == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto audios_obj = std::make_unique<RawAudiosObject>();
  audios_obj->audios_ = std::make_unique<AudioRawData[]>(num_audios);
  for (size_t i = 0; i < num_audios; ++i) {
    audios_obj->audios_[i].resize(sizes[i]);
    std::copy_n(static_cast<const std::byte*>(data[i]), sizes[i], audios_obj->audios_[i].data());
  }

  return {};
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

extError_t ORTX_API_CALL OrtxFeatureExtraction(OrtxFeatureExtractor* extractor, OrtxRawAudios* raw_audios,
                                               OrtxTensorResult** result) {
  if (extractor == nullptr || raw_audios == nullptr || result == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto extractor_ptr = static_cast<SpeechFeatureExtractor*>(extractor);
  auto audios_obj = static_cast<RawAudiosObject*>(raw_audios);

  auto result_ptr = std::make_unique<TensorResult>();
  ReturnableStatus status = extractor_ptr->Preprocess(
    ort_extensions::span(audios_obj->audios_.get(), audios_obj->num_audios_), *result_ptr);
  if (status.IsOk()) {
    *result = static_cast<OrtxTensorResult*>(result_ptr.release());
  } else {
    *result = nullptr;
  }

  return status.Code();
}

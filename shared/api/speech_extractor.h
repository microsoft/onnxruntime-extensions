// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_extractor.h"
#include "c_api_utils.hpp"
#include "runner.hpp"


namespace ort_extensions {

typedef std::vector<std::byte> AudioRawData;

class SpeechFeatureExtractor : public OrtxObjectImpl {
 public:
  SpeechFeatureExtractor();

  virtual ~SpeechFeatureExtractor() = default;

 public:
  OrtxStatus Init(std::string_view extractor_def);

  OrtxStatus DoCall(ort_extensions::span<AudioRawData> raw_speech, std::unique_ptr<ortc::Tensor<float>>& log_mel) const;

  static Operation::KernelRegistry kernel_registry_;

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  ortc::IAllocator* allocator_;
};

}  // namespace ort_extensions

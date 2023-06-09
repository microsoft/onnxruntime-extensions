// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct AzureAudioInvoker : public BaseKernel {
  AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(std::string_view auth_token,
               const ortc::Tensor<int8_t>& raw_audio_data,
               ortc::Tensor<std::string>& text);

 private:
  std::string model_uri_;
  std::string model_name_;
  bool verbose_;
};
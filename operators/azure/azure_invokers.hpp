// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct AzureAudioInvoker : public BaseKernel {
  AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& auth_token,
               const ortc::Tensor<uint8_t>& raw_audio_data,
               ortc::Tensor<std::string>& text);

 private:
  std::string model_uri_;
  std::string model_name_;
  bool verbose_;
};

#if ORT_API_VERSION >= 14
struct AzureTritonInvoker : public BaseKernel {
  AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Variadic& inputs,
               ortc::Variadic& outputs);

 private:
  std::string model_uri_;
  std::string model_name_;
  std::string model_ver_;
  std::string verbose_;
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};
#endif

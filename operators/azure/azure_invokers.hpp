// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

// struct AzureAudioInvoker : public BaseKernel {
//   AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info);
//   void Compute(const ortc::Tensor<std::string>& auth_token,
//                const ortc::Tensor<uint8_t>& raw_audio_data,
//                ortc::Tensor<std::string>& text);
//
//  private:
//   std::string model_uri_;
//   std::string model_name_;
//   bool verbose_;
// };

struct AzureInvoker : public BaseKernel {
  AzureInvoker(const OrtApi& api, const OrtKernelInfo& info);

 protected:
  ~AzureInvoker() = default;
  std::string model_uri_;
  std::string model_name_;
  std::string model_ver_;
  std::string verbose_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

struct AzureAudioInvoker : public AzureInvoker {
  AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Variadic& inputs, ortc::Tensor<std::string>& output);

 private:
  std::string binary_type_;
};

struct AzureTextInvoker : public AzureInvoker {
  AzureTextInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(std::string_view auth, std::string_view input, ortc::Tensor<std::string>& output);

 private:
  std::string binary_type_;
};

struct AzureTritonInvoker : public AzureInvoker {
  AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs);

 private:
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};

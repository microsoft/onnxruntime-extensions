// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ocos.h"
#include "azure_invoker.hpp"

namespace ort_extensions {
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

}  // namespace ort_extensions

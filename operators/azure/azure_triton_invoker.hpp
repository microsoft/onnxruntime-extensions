// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "azure_invoker.hpp"

namespace ort_extensions {

struct AzureTritonInvoker : public AzureInvoker {
  AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs);

 private:
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};
}  // namespace ort_extensions

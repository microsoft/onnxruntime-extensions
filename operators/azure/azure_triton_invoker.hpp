// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cloud_base_kernel.hpp"
#include "http_client.h"  // triton

namespace ort_extensions {

struct AzureTritonInvoker : public CloudBaseKernel {
  AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) const;

 private:
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};
}  // namespace ort_extensions

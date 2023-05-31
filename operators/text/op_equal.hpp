// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringEqual : BaseKernel {
  KernelStringEqual(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context,
               const ortc::Tensor<std::string>&,
               const ortc::Tensor<std::string>&,
               ortc::Tensor<bool>& output);
};

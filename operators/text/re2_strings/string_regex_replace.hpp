// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringRegexReplace : BaseKernel {
  KernelStringRegexReplace(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               std::string_view str_pattern,
               std::string_view str_rewrite,
               ortc::Tensor<std::string>& output);

 protected:
  int64_t global_replace_;
};
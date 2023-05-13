// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringECMARegexReplace : BaseKernel {
  KernelStringECMARegexReplace(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               std::string_view pattern,
               std::string_view rewrite,
               ortc::Tensor<std::string>& output);

 protected:
  bool global_replace_;
  bool ignore_case_;
};

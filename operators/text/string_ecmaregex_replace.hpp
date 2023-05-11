// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringECMARegexReplace : BaseKernel {
  KernelStringECMARegexReplace(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               const std::string& pattern,
               const std::string& rewrite,
               ortc::Tensor<std::string>& output);

 protected:
  bool global_replace_;
  bool ignore_case_;
};

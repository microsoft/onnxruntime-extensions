// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringECMARegexReplace {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);
  OrtStatusPtr Compute(const ortc::Tensor<std::string>& input,
               std::string_view pattern,
               std::string_view rewrite,
               ortc::Tensor<std::string>& output) const;

 protected:
  int64_t global_replace_{1};
  int64_t ignore_case_{0};
};

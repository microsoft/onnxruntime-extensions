// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include <unordered_map>

struct KernelStringMapping : BaseKernel {
  KernelStringMapping(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::TensorT<std::string>& input,
               ortc::TensorT<std::string>& output);

 private:
  std::unordered_map<std::string, std::string> map_;
};

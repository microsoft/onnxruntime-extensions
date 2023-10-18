// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include <unordered_map>

struct KernelStringMapping {

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);
  OrtStatusPtr Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<std::string>& output) const;

 private:
  std::unordered_map<std::string, std::string> map_;
};

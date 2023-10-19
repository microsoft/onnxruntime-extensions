// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

OrtStatusPtr RaggedTensorToSparse(const ortc::Tensor<int64_t>& n_element,
                                  ortc::Tensor<int64_t>& output_0,
                                  ortc::Tensor<int64_t>& output_1);

struct KernelRaggedTensoroDense {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    return OrtW::GetOpAttribute(info, "missing_value", missing_value_);
  }

  OrtStatusPtr Compute(const ortc::Tensor<int64_t>& input0,
                       const ortc::Tensor<int64_t>& input1,
                       const ortc::Tensor<int64_t>& input2,
                       const ortc::Tensor<int64_t>& input3,
                       ortc::Tensor<int64_t>& output) const;

 private:
  int64_t missing_value_{-1};
};

OrtStatusPtr StringRaggedTensorToDense(const ortc::Tensor<int64_t>& input0,
                      const ortc::Tensor<std::string>& input1,
                      const ortc::Tensor<int64_t>& input2,
                      const ortc::Tensor<std::string>& input3,
                      ortc::Tensor<std::string>& output);

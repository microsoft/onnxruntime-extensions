// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "op_equal.hpp"
#include "op_equal_impl.hpp"
#include <string>

KernelStringEqual::KernelStringEqual(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
}

void KernelStringEqual::Compute(OrtKernelContext* context,
                                const ortc::TensorT<std::string>&,
                                const ortc::TensorT<std::string>&,
                                ortc::TensorT<bool>& output) {
  KernelEqual_Compute<std::string>(api_, ort_, context);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct KernelRaggedTensoroSparse : BaseKernel {
  KernelRaggedTensoroSparse(const OrtApi& api, const OrtKernelInfo& info)
      : BaseKernel(api, info) {}

  void Compute(const ortc::Tensor<int64_t>& n_element,
               ortc::Tensor<int64_t>& output_0,
               ortc::Tensor<int64_t>& output_1);
};

struct CommonRaggedTensoroDense : BaseKernel {
  CommonRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info);

 protected:
  void GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims);
  int64_t GetMaxCol(int64_t n, const int64_t* p_indices);
};

struct KernelRaggedTensoroDense : CommonRaggedTensoroDense {
  KernelRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<int64_t>& input0,
               const ortc::Tensor<int64_t>& input1,
               const ortc::Tensor<int64_t>& input2,
               const ortc::Tensor<int64_t>& input3,
               ortc::Tensor<int64_t>& output);

 private:
  int64_t missing_value_;
};

struct KernelStringRaggedTensoroDense : CommonRaggedTensoroDense {
  KernelStringRaggedTensoroDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<int64_t>& input0,
               const ortc::Tensor<std::string>& input1,
               const ortc::Tensor<int64_t>& input2,
               const ortc::Tensor<std::string>& input3,
               ortc::Tensor<std::string>& output);
};

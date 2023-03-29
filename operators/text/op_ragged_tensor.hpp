// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct KernelRaggedTensorToSparse : BaseKernel {
  KernelRaggedTensorToSparse(const OrtApi& api, const OrtKernelInfo& info)
      : BaseKernel(api, info) {}

  void Compute(const ortc::TensorT<int64_t>& n_element,
               ortc::TensorT<int64_t>& output_0,
               ortc::TensorT<int64_t>& output_1);
};

struct CommonRaggedTensorToDense : BaseKernel {
  CommonRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);

 protected:
  void GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims);
  int64_t GetMaxCol(int64_t n, const int64_t* p_indices);
};

struct KernelRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::TensorT<int64_t>& input0,
               const ortc::TensorT<int64_t>& input1,
               const ortc::TensorT<int64_t>& input2,
               const ortc::TensorT<int64_t>& input3,
               ortc::TensorT<int64_t>& output);

 private:
  int64_t missing_value_;
};

struct KernelStringRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelStringRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::TensorT<int64_t>& input0,
               const ortc::TensorT<std::string>& input1,
               const ortc::TensorT<int64_t>& input2,
               const ortc::TensorT<std::string>& input3,
               ortc::TensorT<std::string>& output);
};

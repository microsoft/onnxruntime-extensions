// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

struct KernelRaggedTensorToSparse : BaseKernel {
  KernelRaggedTensorToSparse(const OrtApi& api, const OrtKernelInfo& info)
      : BaseKernel(api, info) {}

  void Compute(OrtKernelContext* context);
};

struct CustomOpRaggedTensorToSparse : OrtW::CustomOpBase<CustomOpRaggedTensorToSparse, KernelRaggedTensorToSparse> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

struct CommonRaggedTensorToDense : BaseKernel {
  CommonRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);

 protected:
  void GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims);
  int64_t GetMaxCol(int64_t n, const int64_t* p_indices);
};

struct KernelRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  int64_t missing_value_;
};

struct CustomOpRaggedTensorToDense : OrtW::CustomOpBase<CustomOpRaggedTensorToDense, KernelRaggedTensorToDense> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

struct KernelStringRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelStringRaggedTensorToDense(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringRaggedTensorToDense : OrtW::CustomOpBase<CustomOpStringRaggedTensorToDense,
                                                              KernelStringRaggedTensorToDense> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

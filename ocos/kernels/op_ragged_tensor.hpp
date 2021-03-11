// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils/string_utils.h"

struct KernelRaggedTensorToSparse : BaseKernel {
  KernelRaggedTensorToSparse(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpRaggedTensorToSparse : Ort::CustomOpBase<CustomOpRaggedTensorToSparse, KernelRaggedTensorToSparse> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

struct CommonRaggedTensorToDense : BaseKernel {
  CommonRaggedTensorToDense(OrtApi api, const OrtKernelInfo* info);

 protected:
  void GetInputDims(OrtKernelContext* context, const OrtValue** inputs, OrtTensorDimensions* dims);
  int64_t GetMaxCol(int64_t n, const int64_t* p_indices);
};

struct KernelRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelRaggedTensorToDense(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  int64_t missing_value_;
};

struct CustomOpRaggedTensorToDense : Ort::CustomOpBase<CustomOpRaggedTensorToDense, KernelRaggedTensorToDense> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

struct KernelStringRaggedTensorToDense : CommonRaggedTensorToDense {
  KernelStringRaggedTensorToDense(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringRaggedTensorToDense : Ort::CustomOpBase<CustomOpStringRaggedTensorToDense, KernelStringRaggedTensorToDense> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

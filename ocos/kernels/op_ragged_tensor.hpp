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

struct KernelRaggedTensorToDense : BaseKernel {
  KernelRaggedTensorToDense(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpRaggedTensorToDense : Ort::CustomOpBase<CustomOpRaggedTensorToDense, KernelRaggedTensorToDense> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

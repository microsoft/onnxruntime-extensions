// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils/string_utils.h"

struct KernelStringLower : BaseKernel {
  KernelStringLower(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringLower : Ort::CustomOpBase<CustomOpStringLower, KernelStringLower> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

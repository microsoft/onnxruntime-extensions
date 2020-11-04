// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "../utils.h"

struct KernelStringReplace : BaseKernel {
  KernelStringReplace(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringReplace : Ort::CustomOpBase<CustomOpStringReplace, KernelStringReplace> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

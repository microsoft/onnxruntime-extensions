// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils/string_utils.h"

struct KernelStringEqual : BaseKernel {
  KernelStringEqual(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringEqual : Ort::CustomOpBase<CustomOpStringEqual, KernelStringEqual> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

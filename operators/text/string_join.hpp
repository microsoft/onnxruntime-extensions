// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringJoin : BaseKernel {
  KernelStringJoin(const OrtApi& api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringJoin : Ort::CustomOpBase<CustomOpStringJoin, KernelStringJoin> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

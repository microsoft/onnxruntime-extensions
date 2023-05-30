// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringStrip : BaseKernel {
  KernelStringStrip(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringStrip : OrtW::CustomOpBase<CustomOpStringStrip, KernelStringStrip> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

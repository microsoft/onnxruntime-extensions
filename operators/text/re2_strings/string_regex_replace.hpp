// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringRegexReplace : BaseKernel {
  KernelStringRegexReplace(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 protected:
  int64_t global_replace_;
};

struct CustomOpStringRegexReplace : Ort::CustomOpBase<CustomOpStringRegexReplace, KernelStringRegexReplace> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

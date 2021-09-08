// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringECMARegexReplace : BaseKernel {
  KernelStringECMARegexReplace(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 protected:
  bool global_replace_;
  bool ignore_case_;
};

struct CustomOpStringECMARegexReplace : Ort::CustomOpBase<CustomOpStringECMARegexReplace, KernelStringECMARegexReplace> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelSegmentExtraction : BaseKernel {
  KernelSegmentExtraction(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpSegmentExtraction : Ort::CustomOpBase<CustomOpSegmentExtraction, KernelSegmentExtraction> {
  size_t GetInputTypeCount() const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
  const char* GetName() const;
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
};

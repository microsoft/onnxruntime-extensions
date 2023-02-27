// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

// See https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/regex_split_with_offsets.md.
struct KernelStringRegexSplitWithOffsets : BaseKernel {
  KernelStringRegexSplitWithOffsets(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringRegexSplitWithOffsets : OrtW::CustomOpBase<CustomOpStringRegexSplitWithOffsets, KernelStringRegexSplitWithOffsets> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

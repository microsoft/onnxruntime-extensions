// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels/kernels.h"
#include "utils.h"

struct KernelStringNormalizer : BaseKernel {
  KernelStringNormalizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
  ~KernelStringNormalizer();

 private:
  void* normalizer_;  // type: sentencepiece::normalizer::Normalizer*
                      // using void* avoids including a header from sentencepiece
};

struct CustomOpStringNormalizer : Ort::CustomOpBase<CustomOpStringNormalizer, KernelStringNormalizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels/kernels.h"
#include "utils.h"
#include "normalizer.h"

struct KernelStringNormalizer : BaseKernel {
  KernelStringNormalizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);
  ~KernelStringNormalizer();

 private:
  sentencepiece::normalizer::Normalizer* normalizer_;
};

struct CustomOpStringNormalizer : Ort::CustomOpBase<CustomOpStringNormalizer, KernelStringNormalizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

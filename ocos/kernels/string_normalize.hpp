// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils.h"
#include "normalizer.h"

struct KernelStringNormalize : BaseKernel {
  KernelStringNormalize(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 protected:
  ~KernelStringNormalize();

 private:
  sentencepiece::normalizer::Normalizer* normalizer_;
};

struct CustomOpStringNormalize : Ort::CustomOpBase<CustomOpStringNormalize, KernelStringNormalize> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

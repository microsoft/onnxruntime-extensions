// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

struct KernelStringHash : BaseKernel {
  KernelStringHash(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringHash : Ort::CustomOpBase<CustomOpStringHash, KernelStringHash> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

struct KernelStringHashFast : BaseKernel {
  KernelStringHashFast(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringHashFast : Ort::CustomOpBase<CustomOpStringHashFast, KernelStringHashFast> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

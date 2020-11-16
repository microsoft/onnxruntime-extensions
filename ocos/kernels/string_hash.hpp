// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels.h"
#include "utils.h"

uint64_t Hash64(const char* data, size_t n, uint64_t seed);

inline uint64_t Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

uint64_t Hash64Fast(const char* data, size_t n);

struct KernelStringHash : BaseKernel {
  KernelStringHash(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpStringHash : Ort::CustomOpBase<CustomOpStringHash, KernelStringHash> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
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
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

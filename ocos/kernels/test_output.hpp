// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "kernels.h"
#include "../utils.h"

#include <vector>
#include <cmath>
#include <algorithm>

struct KernelOne : BaseKernel {
  KernelOne(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

struct KernelTwo : BaseKernel {
  KernelTwo(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

struct KernelNegPos : BaseKernel {
  KernelNegPos(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpNegPos : Ort::CustomOpBase<CustomOpNegPos, KernelNegPos> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info);
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};
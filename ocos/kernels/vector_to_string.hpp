// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "kernels.h"
#include "utils/string_utils.h"

class VectorToStringImplBase
{
 public:
  virtual std::vector<std::string> Compute(const void* input, const OrtTensorDimensions& input_dim, OrtTensorDimensions& output_dim) = 0;
};

struct KernelVectorToString : BaseKernel {
  KernelVectorToString(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  std::shared_ptr<VectorToStringImplBase> impl_;
};

struct CustomOpVectorToString : Ort::CustomOpBase<CustomOpVectorToString, KernelVectorToString> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

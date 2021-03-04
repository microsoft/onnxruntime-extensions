// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "kernels/kernels.h"
#include "utils/string_utils.h"

struct KernelBertTokenizer : BaseKernel {
  KernelBertTokenizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  void Split(const std::u32string& text, std::vector<std::u32string>& words);

  int64_t max_input_chars_per_word_;
  std::u32string suffix_indicator_;
  std::unordered_map<std::u32string, int32_t> vocab_;
};

struct CustomOpBertTokenizer : Ort::CustomOpBase<CustomOpBertTokenizer, KernelBertTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

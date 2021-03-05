// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <vector>
#include "kernels/kernels.h"
#include "kernels/string_common.h"
#include "utils/string_utils.h"

struct KernelBertTokenizer : BaseKernel {
  KernelBertTokenizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
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

void KernelBertTokenizer_Split(const std::u32string& suffix_indicator,
                               const std::u32string& text,
                               std::vector<std::u32string>& words);

void KernelBertTokenizer_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
                                   const std::u32string& suffix_indicator,
                                   const std::vector<ustring>& texts,
                                   std::vector<int32_t>& indices,
                                   std::vector<int64_t>& rows,
                                   int64_t max_input_chars_per_word = 200);

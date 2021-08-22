// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

class BasicTokenizer {
 public:
  BasicTokenizer(bool do_lower_case, bool tokenize_chinese_chars, bool strip_accents, bool tokenize_punctuation, bool remove_control_chars);
  std::vector<std::string> Tokenizer(std::string input);

 private:
  bool do_lower_case_;
  bool strip_accents_;
  bool tokenize_chinese_chars_;
  bool tokenize_punctuation_;
  bool remove_control_chars_;
};

struct KernelBasicTokenizer : BaseKernel {
  KernelBasicTokenizer(OrtApi api);
  void Compute(OrtKernelContext* context);
};

struct CustomOpBasicTokenizer : Ort::CustomOpBase<CustomOpBasicTokenizer, KernelBasicTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

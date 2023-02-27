// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "ustring.h"

class BasicTokenizer {
 public:
  BasicTokenizer(bool do_lower_case, bool tokenize_chinese_chars, bool strip_accents, bool tokenize_punctuation,
                 bool remove_control_chars);
  std::vector<ustring> Tokenize(ustring text);

 private:
  bool do_lower_case_;
  bool strip_accents_;
  bool tokenize_chinese_chars_;
  bool tokenize_punctuation_;
  bool remove_control_chars_;
};

struct KernelBasicTokenizer : BaseKernel {
  KernelBasicTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  std::shared_ptr<BasicTokenizer> tokenizer_;
};

struct CustomOpBasicTokenizer : OrtW::CustomOpBase<CustomOpBasicTokenizer, KernelBasicTokenizer> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

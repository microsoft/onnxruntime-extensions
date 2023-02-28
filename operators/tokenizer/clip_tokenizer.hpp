// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "bpetokenizer.hpp"

struct KernelClipBpeTokenizer : BaseKernel {
  KernelClipBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(OrtKernelContext* context);

 private:
  std::vector<int64_t> Tokenize(ustring& input, int64_t max_length);

  int64_t padding_length_;
  std::list<int> byte_list_;
  std::shared_ptr<VocabData> bbpe_tokenizer_;
};

struct CustomOpClipBpeTokenizer : OrtW::CustomOpBase<CustomOpClipBpeTokenizer, KernelClipBpeTokenizer> {
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

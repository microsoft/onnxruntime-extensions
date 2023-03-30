// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "bpe_tokenizer.hpp"

struct KernelBpeTokenizer : BaseKernel {
  KernelBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::TensorT<std::string>& input,
               ortc::TensorT<int64_t>& tokenize_output,
               ortc::TensorT<int64_t>& attention_mask);

 private:
  std::vector<int64_t> Tokenize(const ustring& input, int64_t max_length);

  int64_t padding_length_;
  std::list<std::pair<int, int>> byte_list_;
  std::shared_ptr<VocabData> bbpe_tokenizer_;
};

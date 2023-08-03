// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "bpe_tokenizer.hpp"

struct KernelRobertaBpeTokenizer : BaseKernel {
  KernelRobertaBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& tokenize_output,
               std::optional<ortc::Tensor<int64_t>*> attention_mask,
               std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;

 private:
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<int64_t> Tokenize(ustring& input, int64_t max_length, bool compute_offset_mapping,
                                std::list<OffsetMappingType>& offset_map) const;

  int64_t padding_length_;
  std::shared_ptr<VocabData> bbpe_tokenizer_;
};

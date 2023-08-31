// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "ustring.h"

#include <string>
#include <vector>

struct BpeModelConf {
  static const char kModel_GPT2[];
  static const char kModel_Roberta[];
  static const char kModel_CLIP[];

  const char* name_{kModel_GPT2};
  const char* unk_token_{"<|endoftext|>"};
  const char* bos_token_{"<|endoftext|>"};
  const char* eos_token_{"<|endoftext|>"};
  const char* pad_token_{nullptr};

  std::string GetSpecialTokens() const;
};

class BpeModel;

struct KernelBpeTokenizer : BaseKernel {
  KernelBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info, const BpeModelConf& conf);

  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& tokenize_output,
               std::optional<ortc::Tensor<int64_t>*> attention_mask,
               std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;

  const char* ModelName() const { return bpe_conf_.name_; }

 protected:
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<int64_t> Tokenize(ustring& input,
                                int64_t max_length,
                                bool compute_offset_mapping,
                                std::list<OffsetMappingType>& offset_map) const;
  int64_t padding_length_;
  std::unique_ptr<BpeModel> bbpe_tokenizer_;
  const BpeModelConf& bpe_conf_;

  uint32_t unk_token_id_{};
  uint32_t bos_token_id_{};
  uint32_t eos_token_id_{};
  uint32_t pad_token_id_{};
};

struct GPT2Tokenizer : KernelBpeTokenizer {
  GPT2Tokenizer(const OrtApi& api, const OrtKernelInfo& info);
};

struct RobertaTokenizer : KernelBpeTokenizer {
  RobertaTokenizer(const OrtApi& api, const OrtKernelInfo& info);
};

struct CLIPTokenizer : KernelBpeTokenizer {
  CLIPTokenizer(const OrtApi& api, const OrtKernelInfo& info);
};

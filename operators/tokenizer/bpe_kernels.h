// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "status.h"
#include "ustring.h"

#include <list>
#include <string>
#include <vector>
#include <functional>

#include "bpe_types.h"

struct BpeModelConf {
  const char* name_{"GPT2"};  // this name may be overridden by the tokenizer's attribute.
  const char* unk_token_{"<|endoftext|>"};
  const char* bos_token_{"<|endoftext|>"};
  const char* eos_token_{"<|endoftext|>"};
  const char* pad_token_{nullptr};

  std::string GetSpecialTokens() const;
};

struct KernelBpeTokenizer {
  KernelBpeTokenizer(const BpeModelConf& conf);
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);

  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;

  const std::string& ModelName() const { return model_name_; }

 protected:
  using OffsetMappingType = std::list<std::pair<size_t, size_t>>;
  std::vector<int64_t> Tokenize(ustring& input,
                                int64_t max_length,
                                bool compute_offset_mapping,
                                std::list<OffsetMappingType>& offset_map) const;

  std::vector<int64_t> SpmTokenize(ustring& input,
                                   int64_t max_length,
                                   bool compute_offset_mapping,
                                   std::list<OffsetMappingType>& offset_map) const;

 protected:
  std::reference_wrapper<BpeModelConf const> bpe_conf_;
  std::string model_name_;
  std::unique_ptr<ort_extensions::BpeModel> bbpe_tokenizer_;

  int64_t padding_length_ = -1;
  uint32_t bos_token_id_{};
  uint32_t eos_token_id_{};
  uint32_t pad_token_id_{};

  std::optional<bool> add_bos_token_;
  std::optional<bool> add_eos_token_;
};

struct GPT2Tokenizer : KernelBpeTokenizer {
  GPT2Tokenizer();
  // required by LiteCustomOp which needs an explicit Compute declaration for non-MSVC compiler.
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
    return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
  }
};

struct RobertaTokenizer : KernelBpeTokenizer {
  RobertaTokenizer();
  // required by LiteCustomOp which needs a explicit Compute declaration for non-MSVC compiler.
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
    return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
  }
};

struct CLIPTokenizer : KernelBpeTokenizer {
  CLIPTokenizer();
  // required by LiteCustomOp which needs a explicit Compute declaration for non-MSVC compiler.
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
    return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
  }
};

struct SpmTokenizer : KernelBpeTokenizer {
  SpmTokenizer();
  // required by LiteCustomOp which needs a explicit Compute declaration for non-MSVC compiler.
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
    return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
  }
};

class JsonFastTokenizer : KernelBpeTokenizer {
 public:
  JsonFastTokenizer();
  OrtxStatus Load(const ort_extensions::bpe::TokenJsonConfig& config);
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;

 public:
  auto& GetAddedTokens() const { return added_tokens_; }
  const ort_extensions::BpeModel& GetEncoder() const { return *bbpe_tokenizer_; }

 private:
  BpeModelConf json_conf_;
  std::vector<ort_extensions::bpe::AddedToken> added_tokens_;
};

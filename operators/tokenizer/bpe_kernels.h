// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <string>
#include <vector>
#include <functional>

#include "tokenizer_common.h"


struct BpeModelConf {
  const char* name_{"GPT2"};  // this name may be overridden by the tokenizer's attribute.
  const char* unk_token_{"<|endoftext|>"};
  const char* bos_token_{"<|endoftext|>"};
  const char* eos_token_{"<|endoftext|>"};
  const char* pad_token_{nullptr};

  bool spm_model_{};
  bool add_dummy_prefix_{};
  std::string GetSpecialTokens() const;
};

struct KernelBpeTokenizer {
  using json = nlohmann::json;
  KernelBpeTokenizer(const BpeModelConf& conf);
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info);

  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping) const;

  const std::string& ModelName() const { return model_name_; }
  uint32_t GetTokenId(const std::string& token) const;
  bool GetAddDummyPrefix() const { return bpe_conf_.get().add_dummy_prefix_; }

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

  void CreateUnicodeByteEncoder();

 protected:
  std::string model_name_;
  std::reference_wrapper<BpeModelConf const> bpe_conf_;
  std::unique_ptr<ort_extensions::BpeModel> bbpe_tokenizer_;

  int64_t padding_length_ = -1;
  uint32_t bos_token_id_{};
  uint32_t eos_token_id_{};
  uint32_t pad_token_id_{};

  std::optional<bool> add_bos_token_;
  std::optional<bool> add_eos_token_;
  std::string unicode_byte_encoder_[256] = {};
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

class JsonFastTokenizer : public KernelBpeTokenizer {
 public:
  JsonFastTokenizer();
  OrtxStatus Load(const ort_extensions::TokenJsonConfig& config);
  OrtxStatus LoadTikTokenBase64(const ort_extensions::TokenJsonConfig& config);
  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask = std::nullopt,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping = std::nullopt) const;

 public:
  const auto& GetAddedTokens() const { return added_tokens_; }
  const ort_extensions::BpeModel& GetEncoder() const { return *bbpe_tokenizer_; }
  bool IsSpmModel() const { return json_conf_.spm_model_; }

 private:
  std::string TokenBytesToString(std::vector<uint8_t>& bytes);
  void LoadSpmModelParams(const json& tok_json);
  void UpdateTokenizer(const ort_extensions::TokenJsonConfig& config, const json& tok_json);

  BpeModelConf json_conf_;
  std::vector<ort_extensions::AddedToken> added_tokens_;
};

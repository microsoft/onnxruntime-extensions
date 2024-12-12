// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <variant>

#include "bpe_kernels.h"
#include "ugm_kernels.hpp"

#include "ext_status.h"
#include "op_def_struct.h"
#include "ort_c_to_cpp.h"

namespace ort_extensions {

class JsonTokenizerOpKernel {
 public:
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    std::string config_json;
    ORTW_RETURN_IF_ERROR(OrtW::API::GetOpAttributeString(api, info, "tokenizer_config", config_json));

    std::string vocab_json;
    ORTW_RETURN_IF_ERROR(OrtW::API::GetOpAttributeString(api, info, "tokenizer_vocab", vocab_json));

    TokenJsonConfig cfg;
    OrtxTokenizerBlob blob({config_json.c_str(), config_json.length()},
                           {vocab_json.c_str(), vocab_json.length()});

    ORTX_RETURN_IF_ERROR(cfg.LoadFromBlob(blob));

    auto type = TokenJsonConfig::GetTokenType(cfg.tokenizer_class_);
    if (type == TokenType::kUnigram) {
      tokenizer_ = std::make_unique<SpmUgmTokenizer>();
    } else if (type == TokenType::kBPE) {
      tokenizer_ = std::make_unique<JsonFastTokenizer>();
    } else {
      return OrtxStatus(kOrtxErrorCorruptData, "Unknown tokenizer type");
    }

    return std::visit([&](auto& ptr) { return ptr->Load(cfg); }, tokenizer_);
  }

  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output,
                     std::optional<ortc::Tensor<int64_t>*> attention_mask = std::nullopt,
                     std::optional<ortc::Tensor<int64_t>*> offset_mapping = std::nullopt) const {

    return std::visit([&](auto& ptr) { 
        return ptr->Compute(input, tokenize_output, attention_mask, offset_mapping); 
    }, tokenizer_);
  }

 private:
  std::variant<std::unique_ptr<JsonFastTokenizer>, std::unique_ptr<SpmUgmTokenizer>> tokenizer_;
};

}  // namespace ort_extensions

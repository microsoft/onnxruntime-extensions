// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <variant>

#include "token_fwd.h"
#include "ugm_kernels.hpp"
#include "bpe_kernels.h"
#include "bpe_jsoncfg.hpp"
#include "bpe_streaming.hpp"
#include "c_api_utils.hpp"

namespace ort_extensions {

class TokenizerImpl : public OrtxObjectImpl {
 public:
  TokenizerImpl();
  virtual ~TokenizerImpl();

 public:
  OrtxStatus Load(const std::string& tok_path);

  OrtxStatus Tokenize(const std::vector<std::string_view>& input, std::vector<std::vector<extTokenId_t>>& t_ids) const {
    return BatchEncode(input, t_ids);
  }

  OrtxStatus Detokenize(const std::vector<span<extTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const {
    return BatchDecode(t_ids, t_text);
  }

  OrtxStatus Token2Id(const std::string& token, extTokenId_t& id) const {
    id = std::visit([&](auto& tokenizer) { return tokenizer->GetTokenId(token); }, tokenizer_);
    return {};
  }

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, std::unique_ptr<BPEDecoderState>& cache) const {
    BPEDecoderState* state_ptr = cache.get();
    OrtxStatus status = Id2Token(id, token, &state_ptr);
    if (status.IsOk()) {
      if (state_ptr != cache.get()) {
        cache.reset(state_ptr);
      }
    }

    return status;
  }

  OrtxStatus BatchEncode(const std::vector<std::string_view>& input,
                         std::vector<std::vector<extTokenId_t>>& t_ids) const;

  OrtxStatus BatchDecode(const std::vector<span<extTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const;

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, BPEDecoderState** state) const;

  OrtxStatus GetDecoderPromptIds(size_t batch_size, const char* lang, const char* task, int no_timestamps,
                                 std::vector<std::vector<extTokenId_t>>& t_ids) const;

 private:
  using bpe_tokenizer_t = std::unique_ptr<JsonFastTokenizer>;
  using ugm_tokenizer_t = std::unique_ptr<SpmUgmTokenizer>;
  std::variant<bpe_tokenizer_t, ugm_tokenizer_t> tokenizer_;

  std::shared_ptr<ort_extensions::bpe::TokenJsonConfig> tok_config_;
  std::unique_ptr<BpeStreamingDecoder> detokenizer_;
};

}  // namespace ort_extensions

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <variant>

#include "bpe_kernels.h"
#include "ugm_kernels.hpp"
#include "tokenizer_jsconfig.hpp"
#include "bpe_streaming.hpp"
#include "c_api_utils.hpp"

namespace ort_extensions {

class TokenizerImpl : public OrtxObjectImpl {
 public:
  TokenizerImpl();
  virtual ~TokenizerImpl();

 public:
  OrtxStatus Load(const std::string& tok_path);
  OrtxStatus Load(const OrtxTokenizerBlob& blob);

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

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, std::unique_ptr<TokenizerDecodingState>& cache) const {
    TokenizerDecodingState* state_ptr = cache.get();
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

  std::string PHI_VISION_CHAT_TEMPLATE;
  std::string PHI3_CHAT_TEMPLATE;
  std::string PHI3_SMALL_CHAT_TEMPLATE;
  std::string PHI3_MEDIUM_CHAT_TEMPLATE;
  std::string PHI3_5_CHAT_TEMPLATE;
  std::string PHI4_CHAT_TEMPLATE;
  std::string LLAMA2_CHAT_TEMPLATE;
  std::string LLAMA3_CHAT_TEMPLATE;
  std::string LLAMA3_2_CHAT_TEMPLATE;
  std::string LLAMA3_3_CHAT_TEMPLATE;
  std::string DEEPSEEK_CHAT_TEMPLATE;

  std::unordered_map<std::string, std::string> model_to_template_map;
  std::string chat_template;
  std::vector<std::unordered_map<std::string, std::string>> messages;

  bool add_generation_prompt;
  std::string eos_token;
  std::string bos_token;
  std::vector<std::string> custom_tools;
  bool tools_in_user_message;
  std::string strftime_now;
  std::string date_string;
  std::vector<std::string> builtin_tools;


  OrtxStatus PhiVisionChatTemplate(std::string& output);

  OrtxStatus Phi3ChatTemplate(std::string& output);

  OrtxStatus Phi3SmallChatTemplate(std::string& output);

  OrtxStatus Phi3MediumChatTemplate(std::string& output);

  OrtxStatus Phi4ChatTemplate(std::string& output);

  OrtxStatus Llama2ChatTemplate(std::string& output);

  OrtxStatus Llama3ChatTemplate(std::string& output);

  OrtxStatus Llama3_2ChatTemplate(std::string& output);

  OrtxStatus Llama3_3ChatTemplate(std::string& output);

  OrtxStatus DeepSeekChatTemplate(std::string& output);
  
  void NormalizeNewlines(std::vector<std::unordered_map<std::string, std::string>>& messages);

  void InitializeChatParameters(
    bool add_prompt = true,
    const std::string& eos = "<|endoftext|>",
    const std::string& bos = "<|startoftext|>",
    const std::vector<std::string>& custom_tools_param = {},
    bool tools_in_user_message_param = true,
    const std::string& strftime_param = "",
    const std::string& date_str = "26 Jul 2024",
    const std::vector<std::string>& builtin_tools_param = {}
);
  
  OrtxStatus ApplyChatTemplate(const std::string model_str, std::vector<std::unordered_map<std::string, std::string>> messages, std::string& output);

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, TokenizerDecodingState** state) const;

  OrtxStatus GetDecoderPromptIds(size_t batch_size, const char* lang, const char* task, int no_timestamps,
                                 std::vector<std::vector<extTokenId_t>>& t_ids) const;

 private:
  OrtxStatus LoadTokenizer(const OrtxTokenizerBlob* blob = nullptr);

  using bpe_tokenizer_t = std::unique_ptr<JsonFastTokenizer>;
  using ugm_tokenizer_t = std::unique_ptr<SpmUgmTokenizer>;
  std::variant<bpe_tokenizer_t, ugm_tokenizer_t> tokenizer_;

  using bpe_decoder_t = std::unique_ptr<BpeStreamingDecoder>;
  using ugm_decoder_t = std::unique_ptr<SpmUgmDecoder>;
  std::variant<bpe_decoder_t, ugm_decoder_t> detokenizer_;

  std::shared_ptr<ort_extensions::TokenJsonConfig> tok_config_;
};

}  // namespace ort_extensions

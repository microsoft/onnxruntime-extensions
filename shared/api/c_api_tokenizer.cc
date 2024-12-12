// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "c_api_utils.hpp"
#include "tokenizer_impl.h"

using namespace ort_extensions;

class DetokenizerCache : public OrtxObjectImpl {
 public:
  DetokenizerCache() : OrtxObjectImpl(extObjectKind_t::kOrtxKindDetokenizerCache) {}
  ~DetokenizerCache() override = default;

  std::unique_ptr<TokenizerDecodingState> decoder_state_{};
  std::string last_text_{};  // last detokenized text
};

template <>
OrtxObject* OrtxObjectFactory::CreateForward<DetokenizerCache>() {
  return Create<DetokenizerCache>();
}

extError_t ORTX_API_CALL OrtxCreateTokenizer(OrtxTokenizer** tokenizer, const char* tokenizer_path) {
  // test if the tokenizer_path is a valid directory
  if (tokenizer_path == nullptr) {
    ReturnableStatus::last_error_message_ = "The tokenizer data directory is null";
    return kOrtxErrorInvalidArgument;
  }

  if (!path(tokenizer_path).is_directory()) {
    ReturnableStatus::last_error_message_ = std::string("Cannot find the directory of ") + tokenizer_path;
    return kOrtxErrorInvalidArgument;
  }

  ReturnableStatus status;
  // auto ptr = ort_extensions::CreateTokenizer(tokenizer_path, "", &status);
  auto ptr = std::make_unique<ort_extensions::TokenizerImpl>();
  status = ptr->Load(tokenizer_path);
  if (status.IsOk()) {
    *tokenizer = static_cast<OrtxTokenizer*>(ptr.release());
    return extError_t();
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxCreateTokenizerFromBlob(OrtxTokenizer** tokenizer, const OrtxTokenizerBlob* blob) {
  // test if the tokenizer_path is a valid directory
  if (blob == nullptr) {
    ReturnableStatus::last_error_message_ = "The tokenizer blob is null";
    return kOrtxErrorInvalidArgument;
  }

  ReturnableStatus status;
  auto ptr = std::make_unique<ort_extensions::TokenizerImpl>();
  status = ptr->Load(*blob);
  if (status.IsOk()) {
    *tokenizer = static_cast<OrtxTokenizer*>(ptr.release());
    return extError_t();
  }

  return status.Code();
}

extError_t ORTX_API_CALL OrtxTokenize(const OrtxTokenizer* tokenizer, const char* input[], size_t batch_size,
                                      OrtxTokenId2DArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status = token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer);
  if (!status.IsOk()) {
    return status.Code();
  }

  std::vector<std::vector<extTokenId_t>> t_ids;
  std::vector<std::string_view> input_view;
  std::transform(input, input + batch_size, std::back_inserter(input_view),
                 [](const char* str) { return std::string_view(str); });

  status = token_ptr->Tokenize(input_view, t_ids);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result = std::make_unique<ort_extensions::TokenId2DArray>().release();
  result->SetTokenIds(std::move(t_ids));
  *output = static_cast<OrtxTokenId2DArray*>(result);

  return extError_t();
}

extError_t ORTX_API_CALL OrtxConvertTokenToId(const OrtxTokenizer* tokenizer, const char* token, extTokenId_t* id) {
  if (tokenizer == nullptr || token == nullptr || id == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }
  auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status = token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer);
  if (!status.IsOk()) {
    return status.Code();
  }

  status = token_ptr->Token2Id(token, *id);
  return status.Code();
}

extError_t ORTX_API_CALL OrtxGetDecoderPromptIds(const OrtxTokenizer* tokenizer, size_t batch_size, const char* lang,
                                                 const char* task, int no_timestamps, OrtxTokenId2DArray** output) {
  if (tokenizer == nullptr || output == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status = token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer);
  if (!status.IsOk()) {
    return status.Code();
  }

  std::vector<std::vector<extTokenId_t>> t_ids;
  status = token_ptr->GetDecoderPromptIds(batch_size, lang, task, no_timestamps, t_ids);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result = std::make_unique<ort_extensions::TokenId2DArray>().release();
  result->SetTokenIds(std::move(t_ids));
  *output = static_cast<OrtxTokenId2DArray*>(result);

  return extError_t();
}

extError_t ORTX_API_CALL OrtxDetokenize(const OrtxTokenizer* tokenizer, const OrtxTokenId2DArray* input,
                                        OrtxStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer));
  if (!status.IsOk()) {
    return status.Code();
  }

  auto input_2d = static_cast<const TokenId2DArray*>(input);
  status = input_2d->IsInstanceOf(extObjectKind_t::kOrtxKindTokenId2DArray);
  if (!status.IsOk()) {
    return status.Code();
  }

  std::vector<span<extTokenId_t const>> t_ids;
  std::transform(input_2d->token_ids().begin(), input_2d->token_ids().end(), std::back_inserter(t_ids),
                 [](const std::vector<extTokenId_t>& vec) { return span<extTokenId_t const>(vec.data(), vec.size()); });

  std::vector<std::string> output_text;
  status = token_ptr->Detokenize(t_ids, output_text);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result = std::make_unique<ort_extensions::StringArray>().release();
  result->SetStrings(std::move(output_text));
  *output = static_cast<OrtxStringArray*>(result);

  return extError_t();
}

extError_t ORTX_API_CALL OrtxDetokenize1D(const OrtxTokenizer* tokenizer, const extTokenId_t* input, size_t len,
                                          OrtxStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer));
  if (!status.IsOk()) {
    return status.Code();
  }

  std::vector<span<extTokenId_t const>> t_ids = {{input, len}};
  std::vector<std::string> output_text;
  status = token_ptr->Detokenize(t_ids, output_text);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result = std::make_unique<ort_extensions::StringArray>().release();
  result->SetStrings(std::move(output_text));
  *output = static_cast<OrtxStringArray*>(result);

  return extError_t();
}

extError_t ORTX_API_CALL OrtxStringArrayGetBatch(const OrtxStringArray* string_array, size_t* length) {
  if (string_array == nullptr || length == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const StringArray*>(string_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindStringArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  *length = token_ptr->strings().size();

  return extError_t();
}

extError_t ORTX_API_CALL OrtxStringArrayGetItem(const OrtxStringArray* string_array, size_t index, const char** item) {
  if (string_array == nullptr || item == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const StringArray*>(string_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindStringArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  if (index >= token_ptr->strings().size()) {
    ReturnableStatus::last_error_message_ = "the index is out of range";
    return kOrtxErrorInvalidArgument;
  }

  *item = token_ptr->strings()[index].c_str();

  return extError_t();
}

extError_t ORTX_API_CALL OrtxTokenId2DArrayGetBatch(const OrtxTokenId2DArray* token_id_2d_array, size_t* length) {
  if (token_id_2d_array == nullptr || length == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_2d_ptr = static_cast<const TokenId2DArray*>(token_id_2d_array);
  ReturnableStatus status(token_2d_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenId2DArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  *length = token_2d_ptr->token_ids().size();

  return extError_t();
}

extError_t ORTX_API_CALL OrtxTokenId2DArrayGetItem(const OrtxTokenId2DArray* token_id_2d_array, size_t index,
                                                   const extTokenId_t** item, size_t* length) {
  if (token_id_2d_array == nullptr || item == nullptr || length == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenId2DArray*>(token_id_2d_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenId2DArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  if (index >= token_ptr->token_ids().size()) {
    ReturnableStatus::last_error_message_ = "the index is out of range";
    return kOrtxErrorInvalidArgument;
  }

  *item = token_ptr->token_ids()[index].data();
  *length = token_ptr->token_ids()[index].size();

  return extError_t();
}

extError_t ORTX_API_CALL OrtxDetokenizeCached(const OrtxTokenizer* tokenizer, OrtxDetokenizerCache* cache,
                                              extTokenId_t next_id,
                                const char** text_out) {
  if (tokenizer == nullptr || cache == nullptr || text_out == nullptr) {
    ReturnableStatus::last_error_message_ = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer));
  if (!status.IsOk()) {
    return status.Code();
  }

  auto cache_ptr = static_cast<DetokenizerCache*>(cache);
  status = ReturnableStatus(cache_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindDetokenizerCache));
  if (!status.IsOk()) {
    return status.Code();
  }

  cache_ptr->last_text_.clear();
  status = ReturnableStatus(token_ptr->Id2Token(next_id, cache_ptr->last_text_, cache_ptr->decoder_state_));
  if (status.IsOk()) {
    *text_out = cache_ptr->last_text_.c_str();
  }

  return status.Code();
}

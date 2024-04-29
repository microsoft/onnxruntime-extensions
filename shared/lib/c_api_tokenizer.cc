// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdarg>
#include <filesystem>
#include <algorithm>

#include "tokenizer_impl.h"

namespace ort_extensions {
class TokenId2DArray : public OrtxObjectImpl {
 public:
  TokenId2DArray() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenId2DArray) {}
  ~TokenId2DArray() override = default;

  void SetTokenIds(std::vector<std::vector<extTokenId_t>>&& token_ids) {
    token_ids_ = token_ids;
  }

  [[nodiscard]] const std::vector<std::vector<extTokenId_t>>& token_ids() const {
    return token_ids_;
  }

 private:
  std::vector<std::vector<extTokenId_t>> token_ids_;
};

class StringArray : public OrtxObjectImpl {
 public:
  StringArray() : OrtxObjectImpl(extObjectKind_t::kOrtxKindStringArray) {}
  ~StringArray() override = default;

  void SetStrings(std::vector<std::string>&& strings) {
    strings_ = strings;
  }

  [[nodiscard]] const std::vector<std::string>& strings() const {
    return strings_;
  }

 private:
  std::vector<std::string> strings_;
};

class DetokenizerCache : public OrtxObjectImpl {
 public:
  DetokenizerCache() : OrtxObjectImpl(extObjectKind_t::kOrtxKindDetokenizerCache) {}
  ~DetokenizerCache() override = default;

  std::unique_ptr<BPEDecoderState> decoder_state_{};
  std::string last_text_{};  // last detokenized text
};

}  // namespace ort_extensions

using namespace ort_extensions;

thread_local std::string last_error_message;

OrtxStatus OrtxObjectImpl::IsInstanceOf(extObjectKind_t kind) const {
  if (ext_kind_ != static_cast<int>(kind)) {
    return {extError_t::kOrtxErrorInvalidArgument,
            "Object is not an instance of the requested type"};
  }
  return {};
}

struct ReturnableStatus {
  ReturnableStatus() = default;
  ReturnableStatus(OrtxStatus&& status) : status_(status) {}
  ~ReturnableStatus() {
    if (!status_.IsOk()) {
      last_error_message = status_.Message();
    }
  }
  ReturnableStatus& operator=(OrtxStatus&& status) {
    status_ = status;
    return *this;
  }
  bool IsOk() const { return status_.IsOk(); }
  extError_t Code() const { return status_.Code(); }

 private:
  OrtxStatus status_;
};

int ORTX_API_CALL OrtxGetAPIVersion() {
  return API_VERSION;
}

const char* OrtxGetLastErrorMessage() {
  return last_error_message.c_str();
}

extError_t ORTX_API_CALL OrtxCreate(extObjectKind_t kind, OrtxObject** object, ...) {
  if (object == nullptr) {
    last_error_message = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  if (kind == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

  va_list args;
  va_start(args, object);

  if (kind == extObjectKind_t::kOrtxKindDetokenizerCache) {
    *object = std::make_unique<DetokenizerCache>().release();
  } else if (kind == extObjectKind_t::kOrtxKindTokenizer) {
    return OrtxCreateTokenizer(static_cast<OrtxTokenizer**>(object), va_arg(args, const char*));
  }

  va_end(args);
  return extError_t();
}

extError_t ORTX_API_CALL OrtxCreateTokenizer(OrtxTokenizer** tokenizer,
                                             const char* tokenizer_path) {
  // test if the tokenizer_path is a valid directory
  if (tokenizer_path == nullptr) {
    last_error_message = "The tokenizer data directory is null";
    return kOrtxErrorInvalidArgument;
  }

  if (!std::filesystem::is_directory(tokenizer_path)) {
    last_error_message = std::string("Cannot find the directory of ") + tokenizer_path;
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

template <typename T>
void Dispose(T* object) {
  auto token_ptr = static_cast<T*>(object);
  std::unique_ptr<T> ptr(token_ptr);
  ptr.reset();
}

extError_t ORTX_API_CALL OrtxDispose(OrtxObject** object) {
  if (object == nullptr || *object == nullptr) {
    return kOrtxErrorInvalidArgument;
  }

  auto Ortx_object = static_cast<OrtxObjectImpl*>(*object);
  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindUnknown) {
    return kOrtxErrorInvalidArgument;
  }

  if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindStringArray) {
    Dispose(static_cast<ort_extensions::StringArray*>(*object));
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenId2DArray) {
    Dispose(static_cast<ort_extensions::TokenId2DArray*>(*object));
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindDetokenizerCache) {
    Dispose(static_cast<ort_extensions::DetokenizerCache*>(*object));
  } else if (Ortx_object->ortx_kind() == extObjectKind_t::kOrtxKindTokenizer) {
    Dispose(static_cast<ort_extensions::TokenizerImpl*>(*object));
  }

  *object = nullptr;
  return extError_t();
}

extError_t ORTX_API_CALL OrtxDisposeOnly(OrtxObject* object) {
  return OrtxDispose(&object);
}

extError_t ORTX_API_CALL OrtxTokenize(const OrtxTokenizer* tokenizer,
                                      const char* input[], size_t batch_size, OrtxTokenId2DArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  auto token_ptr = static_cast<const TokenizerImpl*>(tokenizer);
  ReturnableStatus status =
      token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenizer);
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

extError_t ORTX_API_CALL OrtxDetokenize(const OrtxTokenizer* tokenizer,
                                        const OrtxTokenId2DArray* input, OrtxStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
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
  std::transform(input_2d->token_ids().begin(), input_2d->token_ids().end(),
                 std::back_inserter(t_ids),
                 [](const std::vector<extTokenId_t>& vec) {
                   return span<extTokenId_t const>(vec.data(), vec.size());
                 });

  std::vector<std::string> output_text;
  status = token_ptr->Detokenize(t_ids, output_text);
  if (!status.IsOk()) {
    return status.Code();
  }

  auto result = std::make_unique<ort_extensions::StringArray>().release();
  result->SetStrings(std::move(output_text));
  *output = static_cast<OrtxStringArray*>(result);

  return extError_t();
  ;
}

extError_t ORTX_API_CALL OrtxDetokenize1D(const OrtxTokenizer* tokenizer,
                                          const extTokenId_t* input,
                                          size_t len,
                                          OrtxStringArray** output) {
  if (tokenizer == nullptr || input == nullptr || output == nullptr) {
    last_error_message = "Invalid argument";
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
    last_error_message = "Invalid argument";
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
    last_error_message = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const StringArray*>(string_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindStringArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  if (index >= token_ptr->strings().size()) {
    last_error_message = "the index is out of range";
    return kOrtxErrorInvalidArgument;
  }

  *item = token_ptr->strings()[index].c_str();

  return extError_t();
}

extError_t ORTX_API_CALL OrtxTokenId2DArrayGetBatch(const OrtxTokenId2DArray* token_id_2d_array, size_t* length) {
  if (token_id_2d_array == nullptr || length == nullptr) {
    last_error_message = "Invalid argument";
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

extError_t ORTX_API_CALL OrtxTokenId2DArrayGetItem(const OrtxTokenId2DArray* token_id_2d_array,
                                                   size_t index, const extTokenId_t** item, size_t* length) {
  if (token_id_2d_array == nullptr || item == nullptr || length == nullptr) {
    last_error_message = "Invalid argument";
    return kOrtxErrorInvalidArgument;
  }

  const auto token_ptr = static_cast<const TokenId2DArray*>(token_id_2d_array);
  ReturnableStatus status(token_ptr->IsInstanceOf(extObjectKind_t::kOrtxKindTokenId2DArray));
  if (!status.IsOk()) {
    return status.Code();
  }

  if (index >= token_ptr->token_ids().size()) {
    last_error_message = "the index is out of range";
    return kOrtxErrorInvalidArgument;
  }

  *item = token_ptr->token_ids()[index].data();
  *length = token_ptr->token_ids()[index].size();

  return extError_t();
}

extError_t OrtxDetokenizeCached(const OrtxTokenizer* tokenizer,
                                OrtxDetokenizerCache* cache,
                                extTokenId_t next_id, const char** text_out) {
  if (tokenizer == nullptr || cache == nullptr || text_out == nullptr) {
    last_error_message = "Invalid argument";
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

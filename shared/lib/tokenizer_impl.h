// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ortx_tokenizer.h"
#include "bpe_kernels.h"
#include "bpe_json.hpp"
#include "bpe_streaming.hpp"

namespace ort_extensions {
class OrtxObjectImpl : public OrtxObject {
 public:
  explicit OrtxObjectImpl(extObjectKind_t kind = extObjectKind_t::kOrtxKindUnknown) : OrtxObject() {
    ext_kind_ = static_cast<int>(kind);
  };
  virtual ~OrtxObjectImpl() = default;

  [[nodiscard]] OrtxStatus IsInstanceOf(extObjectKind_t kind) const;
  [[nodiscard]] extObjectKind_t ortx_kind() const {
    if (ext_kind_ < static_cast<int>(extObjectKind_t::kOrtxKindBegin) ||
        ext_kind_ >= static_cast<int>(extObjectKind_t::kOrtxKindEnd)) {
      return extObjectKind_t::kOrtxKindUnknown;
    }
    return static_cast<extObjectKind_t>(ext_kind_);
  }
};

template <typename T>
class span {
 public:
  using value_type = std::remove_cv_t<T>;

  span(T* d, size_t s) : data_(d), size_(s) {}
  span(std::vector<value_type>& v) {
    data_ = v.data();
    size_ = v.size();
  }

  T* data() const { return data_; }
  [[nodiscard]] size_t size() const { return size_; }
  T* begin() const { return data_; }
  T* end() const { return data_ + size_; }

 private:
  T* data_;
  size_t size_;
};

struct BPEDecoderState {
  bool f_special_last{};
  std::string incomplete_utf8_;
};

class TokenizerImpl : public OrtxObjectImpl {
 public:
  TokenizerImpl();
  virtual ~TokenizerImpl();

 public:
  OrtxStatus Load(const std::string& dir);

  OrtxStatus Tokenize(const std::vector<std::string_view>& input,
                      std::vector<std::vector<extTokenId_t>>& t_ids) const {
    return BatchEncode(input, t_ids);
  }

  OrtxStatus Detokenize(const std::vector<span<extTokenId_t const>>& t_ids,
                        std::vector<std::string>& t_text) const {
    return BatchDecode(t_ids, t_text);
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

  OrtxStatus BatchEncode(const std::vector<std::string_view>& input, std::vector<std::vector<extTokenId_t>>& t_ids) const;

  OrtxStatus BatchDecode(const std::vector<span<extTokenId_t const>>& t_ids, std::vector<std::string>& t_text) const;

  OrtxStatus Id2Token(extTokenId_t /* id */, std::string& /* token */, BPEDecoderState** /* state */) const;

 private:
  extTokenId_t eos_token_id_{0};
  std::string tokenizer_dir_;
  std::shared_ptr<ort_extensions::bpe::TokenJsonConfig> tok_config_;
  std::unique_ptr<JsonFastTokenizer> tokenizer_;
  std::unique_ptr<BpeStreamingDecoder> detokenizer_;
};

}  // namespace ort_extensions

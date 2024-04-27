// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "bpe_kernels.h"
#include "bpe_tokenizer.hpp"
#include "bpe_decoder.hpp"

#include "tokenizer_impl.h"

using namespace ort_extensions;

class SimpleAllocator : public ortc::IAllocator {
 public:
  void* Alloc(size_t size) override {
    return malloc(size);
  }

  void Free(void* p) override {
    if (p) {
      free(p);
    }
  }
};

static SimpleAllocator g_allocator;

TokenizerImpl::TokenizerImpl() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenizer){};
TokenizerImpl::~TokenizerImpl(){};

OrtxStatus TokenizerImpl::Load(const std::string& dir) {
  tok_config_ = std::make_shared<ort_extensions::bpe::TokenJsonConfig>();
  auto status = tok_config_->Load(dir);
  if (!status.IsOk()) {
    return status;
  }

  tokenizer_ = std::make_unique<JsonFastTokenizer>();
  // load the tokenizer from a config
  status = tokenizer_->Load(*tok_config_);
  if (status.IsOk()) {
    eos_token_id_ = tokenizer_->GetEncoder().GetTokenId(tok_config_->eos_token_);

    detokenizer_ = std::make_unique<BpeStreamingDecoder>();
    status = detokenizer_->Load(*tok_config_, *tokenizer_);
  }

  return status;
}

OrtxStatus TokenizerImpl::BatchEncode(
    const std::vector<std::string_view>& input,
    std::vector<std::vector<extTokenId_t>>& t_ids) const {
  for (const auto& s : input) {
    ortc::Tensor<int64_t> ts_output(&g_allocator);
    ortc::Tensor<std::string> ts_input = ortc::Tensor<std::string>(std::vector<std::string>{std::string(s)});
    auto status = tokenizer_->Compute(ts_input, ts_output, std::nullopt, std::nullopt);

    if (!status.IsOk()) {
      return status;
    }

    std::vector<extTokenId_t> ids(ts_output.NumberOfElement());
    std::transform(ts_output.Data(), ts_output.Data() + ts_output.NumberOfElement(), ids.begin(),
                   [](int64_t v) { return static_cast<extTokenId_t>(v); });
    t_ids.emplace_back(std::move(ids));
  }

  return {};
}

OrtxStatus TokenizerImpl::BatchDecode(const std::vector<span<extTokenId_t const>>& t_ids,
                                      std::vector<std::string>& t_text) const {
  for (const auto& s : t_ids) {
    std::vector<int64_t> ids(s.size());
    std::transform(s.begin(), s.end(), ids.begin(), [](extTokenId_t v) { return static_cast<int64_t>(v); });
    ortc::Tensor<int64_t> ts_input(std::vector<int64_t>{1, static_cast<int64_t>(ids.size())}, (void*)ids.data());
    ortc::Tensor<std::string> ts_output;
    OrtxStatus status = detokenizer_->Compute(ts_input, ts_output);
    if (!status.IsOk()) {
      return status;
    }
    t_text.emplace_back(ts_output.AsScalar());
  }
  return {};
}

static bool IsSpmTokenizer(const std::string& tok_class) {
  return tok_class == "GemmaTokenizer" || tok_class == "LlamaTokenizer";
}

OrtxStatus TokenizerImpl::Id2Token(extTokenId_t id, std::string& token, BPEDecoderState** state) const {
  auto bpe_state = *state;
  std::unique_ptr<BPEDecoderState> bpe_state_ptr;
  bool is_first = false;
  if (bpe_state == nullptr) {
    bpe_state_ptr = std::make_unique<BPEDecoderState>();
    bpe_state = bpe_state_ptr.get();
    is_first = true;
  }

  bool f_special = false;
  bool& f_special_last = bpe_state->f_special_last;
  auto status = IsSpmTokenizer(tok_config_->tokenizer_class_)
                    ? detokenizer_->SpmId2Token(id, token, f_special)
                    : detokenizer_->Id2Token(id, token, true /* tok_config_.skip_special_tokens_ */, f_special);

  if (status.IsOk()) {
    if (bpe_state_ptr) {
      *state = bpe_state_ptr.release();
    }

    if (tok_config_->clean_up_tokenization_spaces_) {
      if (f_special && (is_first && !f_special_last)) {
        token = std::string(" ") + token;
      }

      if (f_special && id != eos_token_id_) {
        token.push_back(' ');
      }
    }  // end case of whitespace_token_

    if (!bpe_state->incomplete_utf8_.empty()) {
      token = bpe_state->incomplete_utf8_ + token;
      bpe_state->incomplete_utf8_.clear();
    } else {
      if (!token.empty() && ustring::UTF8Len(token.front()) > token.size()) {
        bpe_state->incomplete_utf8_ = token;
        token = "";
      }
    }
  }

  bpe_state->f_special_last = f_special;
  return status;
}

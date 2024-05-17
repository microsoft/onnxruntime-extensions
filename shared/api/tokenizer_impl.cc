// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "bpe_kernels.h"
#include "bpe_tokenizer.hpp"
#include "bpe_decoder.hpp"

#include "tokenizer_impl.h"

using namespace ort_extensions;

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
    detokenizer_ = std::make_unique<BpeStreamingDecoder>();
    status = detokenizer_->Load(tok_config_, *tokenizer_);
  }

  return status;
}

OrtxStatus TokenizerImpl::BatchEncode(
    const std::vector<std::string_view>& input,
    std::vector<std::vector<extTokenId_t>>& t_ids) const {
  for (const auto& s : input) {
    ortc::Tensor<int64_t> ts_output(&CppAllocator::Instance());
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

OrtxStatus TokenizerImpl::Id2Token(extTokenId_t id, std::string& token, BPEDecoderState** state) const {
  return detokenizer_->Id2Token(id, token, state);
}

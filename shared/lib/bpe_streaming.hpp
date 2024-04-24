// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "bpe_kernels.h"
#include "bpe_decoder.hpp"
#include "bpe_json.hpp"
#include "bpe_tokenizer.hpp"

class BpeStreamingDecoder : public KernelBpeDecoder {
 public:
  BpeStreamingDecoder() = default;
  ~BpeStreamingDecoder() override = default;

  // shared the data between the encoder and decoder
  OrtxStatus Load(const ort_extensions::bpe::TokenJsonConfig& tok_config, const JsonFastTokenizer& encoder) {
    bos_token_ = tok_config.bos_token_;
    eos_token_ = tok_config.eos_token_;
    unk_token_ = tok_config.unk_token_;

    auto a_toks = encoder.GetAddedTokens();
    for (const auto& tok : a_toks) {
      added_tokens_[tok.id_] = tok.content_;
      if (tok.token_type_ == "special") {
        all_special_ids_.insert(tok.id_);
      }
    }

    CreateByteDecoder(encoder.GetEncoder());
    arr_vocab_ = encoder.GetEncoder().BuildDecoder();
    whitespace_token_ = tok_config.clean_up_tokenization_spaces_ ? 1 : 0;
    skip_special_tokens_ = 1;
    // en_normalization_ = 0;

    return {};
  }

 private:
  void CreateByteDecoder(const ort_extensions::BpeModel& /* bpe_model */) {
    char32_t index = 256;
    for (char32_t i = 0; i < 256; ++i) {
      /*
      bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
      )
      */
      if ((/* i >= 0 && */ i < 33) || (i >= 127 && i < 161) || (i == 173)) {
        byte_decoder_[index] = gsl::narrow<unsigned char>(i);
      } else {
        byte_decoder_[i] = gsl::narrow<unsigned char>(i);
      }
    }
  }

  std::shared_ptr<ort_extensions::bpe::TokenJsonConfig> tok_config_;
};

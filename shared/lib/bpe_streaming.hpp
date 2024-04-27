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

    auto& a_toks = encoder.GetAddedTokens();
    for (const auto& tok : a_toks) {
      added_tokens_[tok.id_] = tok.content_;
      if (tok.special_) {
        all_special_ids_.insert(tok.id_);
      }
    }

    auto& tok_model = encoder.GetEncoder();
    CreateByteDecoder(tok_model);
    arr_vocab_ = tok_model.BuildDecoder();
    end_of_word_suffix_ = tok_model.GetEndOfWordSuffix();
    whitespace_token_ = tok_config.clean_up_tokenization_spaces_ ? 1 : 0;
    skip_special_tokens_ = 1;
    // en_normalization_ = 0;

    return {};
  }

  static std::string ReplaceAll(std::string_view s, const std::string& search, const std::string& replace) {
    std::string result;
    for (size_t pos = 0;; pos += search.length()) {
      auto new_pos = s.find(search, pos);
      if (new_pos == std::string::npos) {
        result += s.substr(pos, s.size() - pos);
        break;
      }
      result += s.substr(pos, new_pos - pos);
      result += replace;
      pos = new_pos;
    }

    return result;
  }

  static bool IsSpmByteWord(std::string_view word) {
    return word.size() == 6 && word[0] == '<' && word[1] == '0' && word[2] == 'x' && word[5] == '>';
  }

  OrtxStatus Id2Token(extTokenId_t id,
                      std::string& token,
                      bool skip_special_tokens,
                      bool& f_special_last) const {
    bool f_special = all_special_ids_.count(id) ? true : false;
    if (skip_special_tokens && f_special) {
      f_special_last = f_special;
      return {};
    }

    if (added_tokens_.count(id)) {
      const std::string ws = added_tokens_.at(id);
      token = (std::string)ws;
    } else if (static_cast<size_t>(id) < arr_vocab_.size()) {
      const auto str = ustring(arr_vocab_[id]);
      for (auto wchr : str) {
        if (byte_decoder_.count(wchr) == 0 && wchr <= 0xFF) {
          token.push_back(gsl::narrow<unsigned char>(wchr));
        } else {
          unsigned char uchr = byte_decoder_.at(wchr);
          token.push_back(uchr);
        }
      }
    } else {
      if (skip_special_tokens) {
        f_special_last = f_special;
        return {};
      } else {
        token = unk_token_;
      }
    }

    // remove the end_of_word_suffix like </w> or </s> etc.
    if (end_of_word_suffix_.size() > 0) {
      if (token.size() >= end_of_word_suffix_.size() &&
          token.substr(token.size() - end_of_word_suffix_.size()) == end_of_word_suffix_) {
        token = token.substr(0, token.size() - end_of_word_suffix_.size());
        token += ' ';
      }
    }

    f_special_last = f_special;
    return {};
  }

  OrtxStatus SpmId2Token(extTokenId_t id, std::string& token, bool& f_special_last) const {
    const char spm_underscore[] = "\xe2\x96\x81";

    std::string piece = id < arr_vocab_.size() ? arr_vocab_[id] : "";
    bool f_special = false;
    if (piece.empty() || all_special_ids_.count(id)) {
      token = "";
      f_special = true;
    } else if (IsSpmByteWord(piece)) {
      char buf[3] = {piece[3], piece[4], 0};  // something like <0x20>
      token = {static_cast<char>(strtol(buf, NULL, 16))};
    } else {
      token = ReplaceAll(piece, spm_underscore, " ");
    }

    if (!token.empty() && token[0] == ' ' && f_special_last /* && add_dummpy_prefix_ */) {
      token = token.substr(1);
    }

    f_special_last = f_special;
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
        byte_decoder_[index++] = gsl::narrow<unsigned char>(i);
      } else {
        byte_decoder_[i] = gsl::narrow<unsigned char>(i);
      }
    }
  }

  std::shared_ptr<ort_extensions::bpe::TokenJsonConfig> tok_config_;
};

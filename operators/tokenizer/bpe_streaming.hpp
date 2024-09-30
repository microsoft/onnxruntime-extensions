// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "bpe_kernels.h"
#include "bpe_decoder.hpp"
#include "tokenizer_jsconfig.hpp"
#include "bpe_tokenizer_model.hpp"

class BpeStreamingDecoder : public KernelBpeDecoder {
 public:
  BpeStreamingDecoder() = default;
  ~BpeStreamingDecoder() override = default;

  using BPEDecoderState = ort_extensions::TokenizerDecodingState;

  // shared the data between the encoder and decoder
  OrtxStatus Load(
      std::shared_ptr<ort_extensions::TokenJsonConfig const> ptr_config,
      const JsonFastTokenizer& encoder) {
    const auto& tok_config = *ptr_config;
    bos_token_ = tok_config.bos_token_;
    eos_token_ = tok_config.eos_token_;
    unk_token_ = tok_config.unk_token_;
    spm_model_ = encoder.IsSpmModel();

    const auto& a_toks = encoder.GetAddedTokens();
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
    // whitespace_token_ = tok_config.clean_up_tokenization_spaces_ ? 1 : 0;
    skip_special_tokens_ = 1;
    // en_normalization_ = 0;
    add_dummy_prefix_ = tok_config.tokenizer_class_ == "LlamaTokenizer" ? 1 : 0;
    eos_token_id_ = encoder.GetEncoder().GetTokenId(tok_config.eos_token_);

    tok_config_ = ptr_config;
    return {};
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

    std::string piece = id < arr_vocab_.size() ? arr_vocab_[id] : "";
    bool f_special = false;
    if (piece.empty() || all_special_ids_.count(id)) {
      token = "";
      f_special = true;
    } else if (IsSpmByteWord(piece)) {
      char buf[3] = {piece[3], piece[4], 0};  // something like <0x20>
      token = {static_cast<char>(strtol(buf, NULL, 16))};
    } else {
      token = ReplaceAll(piece, std::string(ort_extensions::spm_escaped_space), " ");
    }

    if (!token.empty() && token[0] == ' ' && f_special_last && add_dummy_prefix_) {
      token = token.substr(1);
    }

    f_special_last = f_special;
    return {};
  }

  OrtxStatus Id2Token(extTokenId_t id, std::string& token, BPEDecoderState** state) const {
    auto bpe_state = *state;
    std::unique_ptr<BPEDecoderState> bpe_state_ptr;
    bool is_first = false;
    if (bpe_state == nullptr) {
      bpe_state_ptr = std::make_unique<BPEDecoderState>();
      bpe_state = bpe_state_ptr.get();
      is_first = true;
    }

    bool f_special = bpe_state->f_special_last;  // [Spm]Id2Token needs the last state
    bool f_special_last = bpe_state->f_special_last;
    auto status = spm_model_
                  ? SpmId2Token(id, token, f_special)
                  : Id2Token(id, token, true /* tok_config_.skip_special_tokens_ */, f_special);

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

      bpe_state->incomplete_utf8_ += token;
      token.clear();
      std::string& s_utf8 = bpe_state->incomplete_utf8_;
      size_t utf8_len = 1;
      size_t utf8_all_len = 0;
      for (size_t i = 0; i < s_utf8.size(); i += utf8_len) {
        utf8_len = ustring::UTF8Len(s_utf8[i]);
        if (utf8_len <= s_utf8.size() - i) {
          utf8_all_len += utf8_len;
          auto _t = s_utf8.substr(i, utf8_len);
          token += ustring::ValidateUTF8(_t) > 0 ? _t : "";
        }
      }

      s_utf8 = s_utf8.substr(utf8_all_len);
    }

    bpe_state->f_special_last = f_special;
    return status;
  }

  OrtxStatus Compute(const ortc::Tensor<int64_t>& ids,
                     ortc::Tensor<std::string>& output) const {
    const int64_t* p_ids = ids.Data();
    const auto& ids_dim = ids.Shape();
    std::vector<int64_t> output_dim = {1};
    if (ids_dim.size() > 1) {
      output_dim.resize(ids_dim.size() - 1);
      std::copy(ids_dim.begin(), ids_dim.begin() + ids_dim.size() - 1, output_dim.begin());
    }

    size_t seq_len = ids_dim.back();
    size_t string_batch = ids.NumberOfElement() / seq_len;
    std::vector<std::string> decoded_strings;
    decoded_strings.reserve(string_batch);

    for (auto n = string_batch; n > 0; n--) {
      bool f_special_last = false;
      std::string text;

      for (size_t tok_idx = 0; tok_idx < seq_len; ++tok_idx) {
        const auto id = ort_extensions::narrow<extTokenId_t>(*(p_ids + tok_idx));
        std::string decoded_token;
        auto status = spm_model_
                      ? SpmId2Token(id, decoded_token, f_special_last)
                      : Id2Token(id, decoded_token, true, f_special_last);

        if (!status.IsOk()) {
          return status;
        }

        bool f_special = all_special_ids_.count(id) ? true : false;

        if (whitespace_token_ && f_special && (tok_idx > 0 && !f_special_last)) {
          text.push_back(' ');
        }

        text.append(decoded_token);

        if (whitespace_token_ && f_special && tok_idx != seq_len - 1) {
          text.push_back(' ');
        }
      }

      if (tok_config_->tokenizer_class_.find("CLIP") == 0 && !text.empty() && text.back() == ' ') {
        text.pop_back();
      }

      ptrdiff_t z = ustring::ValidateUTF8(text);
      if (z <= 0) {
        text = text.substr(0, -z);
        text += "\ufffd";  // bad utf-8 string
      }

      decoded_strings.emplace_back(std::move(text));
      p_ids += seq_len;
    }

    output.SetStringOutput(decoded_strings, output_dim);
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

 private:

  extTokenId_t eos_token_id_{0};
  bool add_dummy_prefix_ = false;
  bool spm_model_{};
  std::shared_ptr<ort_extensions::TokenJsonConfig const> tok_config_;
};

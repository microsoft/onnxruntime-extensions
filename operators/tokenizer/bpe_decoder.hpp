// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <set>
#include <map>
#include <unordered_map>
#include <algorithm>
#include <sstream>

#include "ustring.h"
#include "narrow.h"
#include "tokenizer_common.h"

struct KernelBpeDecoder {
 public:
  virtual ~KernelBpeDecoder() = default;

  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    // note: if the attribute doesn't exist in op node, GetOpAttribute doesn't return a failed status;
    std::string vocab;
    OrtStatusPtr status = OrtW::GetOpAttribute(info, "id_vocab", vocab);
    if (status != nullptr || vocab.empty()) {
      if (status == nullptr) {
        status = OrtW::CreateStatus("[BPEDecoder]id vocab text cannot be empty.", ORT_INVALID_ARGUMENT);
      }
      return status;
    }
    BuildIdVocab(vocab);

    std::string byte_decoder;
    status = OrtW::GetOpAttribute(info, "byte_decoder", byte_decoder);
    if (status != nullptr || byte_decoder.empty()) {
      if (status == nullptr) {
        status = OrtW::CreateStatus("[BPEDecoder]byte_decoder cannot be empty.", ORT_INVALID_ARGUMENT);
      }
      return status;
    } else {
      auto um = ParseId2String(byte_decoder);
      std::transform(um.begin(), um.end(), std::inserter(byte_decoder_, byte_decoder_.end()), [](const auto& p) {
        return std::make_pair(static_cast<char32_t>(p.first),
                              ort_extensions::narrow<unsigned char>(std::stoul(p.second)));
      });
    }

    std::string added_tokens;
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "added_tokens", added_tokens));
    if (!added_tokens.empty()) {
      auto um = ParseId2String(added_tokens);
      added_tokens_ = std::map<int64_t, std::string>(um.begin(), um.end());
    }

    std::string all_special_ids;
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "all_special_ids", all_special_ids));
    if (!all_special_ids.empty()) {
      auto um = ParseId2String(all_special_ids);
      std::transform(um.begin(), um.end(), std::inserter(all_special_ids_, all_special_ids_.end()),
                     [](const auto& p) { return p.first; });
    }

    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "en_normalization", en_normalization_));
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "skip_special_tokens", skip_special_tokens_));
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "whitespace_token", whitespace_token_));
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "bos_token", bos_token_));
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "eos_token", eos_token_));
    ORTW_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "unk_token", unk_token_));

    return status;
  }

  std::unordered_map<int64_t, std::string> ParseId2String(const std::string& s_attr) {
    std::unordered_map<int64_t, std::string> result;
    result.reserve(s_attr.size() / 4);
    std::stringstream ss(s_attr);

    std::string line;
    std::string token;
    while (std::getline(ss, line, '\n')) {
      size_t pos_end = 0;
      int64_t v = std::stoll(line, &pos_end);
      if (pos_end >= line.size() || line[pos_end] != '\t') {
        token.clear();
      } else {
        token = line.substr(pos_end + 1);
      }
      result.emplace(v, token);
    }

    return result;
  }

  void BuildIdVocab(const std::string& vocab) {
    arr_vocab_.reserve(vocab.size() / 2);  // give a rough estimation.

    std::string_view v_vocab(vocab);
    size_t last_pos = 0;

    auto c_count = v_vocab.size();
    for (size_t n = 0; n < c_count; ++n) {
      if (v_vocab[n] == '\n') {
        std::string_view s_tok = v_vocab.substr(last_pos, n - last_pos);
        arr_vocab_.emplace_back(std::string(s_tok));
        last_pos = n + 1;
      } else if (n == c_count - 1) {
        std::string_view s_tok = v_vocab.substr(last_pos, n - last_pos + 1);
        arr_vocab_.emplace_back(std::string(s_tok));
      }
    }

    arr_vocab_.shrink_to_fit();
  }

  static bool IsSpmByteWord(std::string_view word) {
    return word.size() == 6 && word[0] == '<' && word[1] == '0' && word[2] == 'x' && word[5] == '>';
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

  OrtxStatus Compute(const ortc::Tensor<int64_t>& ids, ortc::Tensor<std::string>& output) const {
    const int64_t* p_ids = ids.Data();
    const auto& ids_dim = ids.Shape();
    std::vector<int64_t> output_dim = {1};
    if (ids_dim.size() > 1) {
      output_dim.resize(ids_dim.size() - 1);
      std::copy(ids_dim.begin(), ids_dim.begin() + ids_dim.size() - 1, output_dim.begin());
    }

    bool spm_mode = byte_decoder_.count(ustring(ort_extensions::spm_escaped_space)[0]) > 0;

    size_t seq_len = ids_dim.back();
    size_t string_batch = ids.NumberOfElement() / seq_len;
    std::vector<std::string> decoded_strings;
    decoded_strings.reserve(string_batch);

    for (auto n = string_batch; n > 0; n--) {
      std::string text;
      bool f_special_last = false;
      bool f_special = false;
      auto count = static_cast<size_t>(ids.NumberOfElement());

      for (size_t tok_idx = 0; tok_idx < count; ++tok_idx) {
        const auto token = *(p_ids + tok_idx);
        std::string decoded_token;
        f_special = all_special_ids_.count(token) ? true : false;
        if (skip_special_tokens_ && f_special) {
          f_special_last = f_special;
          continue;
        }

        if (added_tokens_.count(token)) {
          const std::string& ws = added_tokens_.at(token);
          decoded_token.assign(ws);
        } else if (static_cast<size_t>(token) < arr_vocab_.size()) {
          const auto piece = arr_vocab_[token];
          if (spm_mode) {
            // sentencepiece case, which doesn't really have a byte decoder
            if ((IsSpmByteWord(piece))) {
              char buf[3] = {piece[3], piece[4], 0};  // something like <0x20>
              char token = {static_cast<char>(strtol(buf, NULL, 16))};
              decoded_token.push_back(token);
            } else {
              decoded_token.append(ReplaceAll(piece, std::string(ort_extensions::spm_escaped_space), " "));
            }
          } else {
            // the common bpe case
            const auto str = ustring(piece);
            for (auto wchr : str) {
              if (byte_decoder_.count(wchr) == 0) {
                if (wchr <= char32_t(0xFF)) {
                  decoded_token.push_back(static_cast<char>(wchr));
                  continue;
                }
                if (skip_special_tokens_) {
                  continue;
                } else {
                  decoded_token = unk_token_;
                  break;
                }
              }
              char uchr = byte_decoder_.at(wchr);
              decoded_token.push_back(uchr);
            }
          }
        } else {
          if (skip_special_tokens_) {
            continue;
          } else {
            decoded_token = unk_token_;
          }
        }
        // remove the end_of_word_suffix like </w> or </s> etc.
        if (end_of_word_suffix_.size() > 0) {
          if (decoded_token.size() >= end_of_word_suffix_.size() &&
              decoded_token.substr(decoded_token.size() - end_of_word_suffix_.size()) == end_of_word_suffix_) {
            decoded_token = decoded_token.substr(0, decoded_token.size() - end_of_word_suffix_.size());
            decoded_token += ' ';
          }
        }

        if (whitespace_token_ && f_special && (tok_idx > 0 && !f_special_last)) {
          text.push_back(' ');
        }

        text.append(decoded_token);

        if (whitespace_token_ && f_special && tok_idx != count - 1) {
          text.push_back(' ');
        }

        f_special_last = f_special;
      }

      decoded_strings.emplace_back(std::move(text));
      p_ids += seq_len;
    }
    output.SetStringOutput(decoded_strings, output_dim);
    return {};
  }

 protected:
  std::string bos_token_{"<|endoftext|>"};
  std::string eos_token_{"<|endoftext|>"};
  std::string unk_token_{"<|endoftext|>"};

  // Since ORT API doesn't support boolean type in ONNX node attribute,
  // all flag attributes here are defined as int64 type to be more explicit.
  int64_t en_normalization_ = 0;
  int64_t skip_special_tokens_ = 0;
  int64_t whitespace_token_ = 0;
  std::vector<std::string> arr_vocab_;
  std::map<char32_t, unsigned char> byte_decoder_;
  std::map<int64_t, std::string> added_tokens_;
  std::set<int64_t> all_special_ids_;
  std::string end_of_word_suffix_;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "file_sys.h"

#include "bpe_kernels.h"
#include "bpe_jsoncfg.hpp"
#include "bpe_tokenizer.hpp"

#include "base64.h"

#include <optional>
#include <limits>

using namespace ort_extensions;

const char kModel_Default[] = "PreTrained";
const char kModel_GPT2[] = "GPT2";
const char kModel_CodeGen[] = "CodeGen";
const char kModel_Roberta[] = "Roberta";
const char kModel_CLIP[] = "CLIP";
const char kModel_Llama[] = "Llama";
const char kModel_Gemma[] = "Gemma";

static bool IsBosEosRequired(const std::string& model_name) {
  return model_name != kModel_GPT2 && model_name != kModel_CodeGen;
}

std::string BpeModelConf::GetSpecialTokens() const {
  std::string special_tokens = unk_token_;  // unk_token_ is required
  auto add_token = [](std::string& sp, const char* tok) {
    if (tok != nullptr) {
      if (sp.find(tok) == std::string::npos) {
        (sp += "\n") += tok;
      }
    }
  };

  add_token(special_tokens, bos_token_);
  add_token(special_tokens, eos_token_);
  add_token(special_tokens, pad_token_);
  return special_tokens;
}

// static
// const unsigned char _Py_ctype_tolower[256] = {
//     0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
//     0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
//     0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
//     0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
//     0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
//     0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
//     0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37,
//     0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
//     0x40, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
//     0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
//     0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
//     0x78, 0x79, 0x7a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
//     0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67,
//     0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
//     0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77,
//     0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
//     0x80, 0x81, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
//     0x88, 0x89, 0x8a, 0x8b, 0x8c, 0x8d, 0x8e, 0x8f,
//     0x90, 0x91, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
//     0x98, 0x99, 0x9a, 0x9b, 0x9c, 0x9d, 0x9e, 0x9f,
//     0xa0, 0xa1, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
//     0xa8, 0xa9, 0xaa, 0xab, 0xac, 0xad, 0xae, 0xaf,
//     0xb0, 0xb1, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7,
//     0xb8, 0xb9, 0xba, 0xbb, 0xbc, 0xbd, 0xbe, 0xbf,
//     0xc0, 0xc1, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7,
//     0xc8, 0xc9, 0xca, 0xcb, 0xcc, 0xcd, 0xce, 0xcf,
//     0xd0, 0xd1, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7,
//     0xd8, 0xd9, 0xda, 0xdb, 0xdc, 0xdd, 0xde, 0xdf,
//     0xe0, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
//     0xe8, 0xe9, 0xea, 0xeb, 0xec, 0xed, 0xee, 0xef,
//     0xf0, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7,
//     0xf8, 0xf9, 0xfa, 0xfb, 0xfc, 0xfd, 0xfe, 0xff,
// };

// Note: the following logic comes from CPython: unicodetype_db.h (_PyUnicode_IsWhitespace)
static bool IsUnicodeSpace(char32_t ch) {
  const std::set<char32_t> unicode_spaces = {
      0x0009,  // CHARACTER TABULATION
      0x000A,  // LINE FEED (LF)
      0x000B,  // LINE TABULATION
      0x000C,  // FORM FEED (FF)
      0x000D,  // CARRIAGE RETURN (CR)
      0x001C,  // FILE SEPARATOR
      0x001D,  // GROUP SEPARATOR
      0x001E,  // RECORD SEPARATOR
      0x001F,  // UNIT SEPARATOR
      0x0020,  // SPACE
      0x0085,  // NEXT
      0x00A0,  // NO-BREAK SPACE
      0x1680,  // OGHAM SPACE MARK
      0x2000,  // EN QUAD
      0x2001,  // EM QUAD
      0x2002,  // EN SPACE
      0x2003,  // EM SPACE
      0x2004,  // THREE-PER-EM SPACE
      0x2005,  // FOUR-PER-EM SPACE
      0x2006,  // SIX-PER-EM SPACE
      0x2007,  // FIGURE SPACE
      0x2008,  // PUNCTUATION SPACE
      0x2009,  // THIN SPACE
      0x200A,  // HAIR SPACE
      0x2028,  // LINE SEPARATOR
      0x2029,  // PARAGRAPH SEPARATOR
      0x202F,  // NARROW NO-BREAK SPACE
      0x205F,  // MEDIUM MATHEMATICAL SPACE
      0x3000,  // IDEOGRAPHIC SPACE
  };
  return unicode_spaces.count(ch) > 0;
}

bool AllSpaceUstring(const ustring& str) {
  return std::all_of(str.begin(), str.end(), [](char32_t ch) { return IsUnicodeSpace(ch); });
}

ustring RemoveConsecutiveSpaces(const ustring& input) {
  ustring result;
  result.reserve(input.size());
  bool lastWasSpace = false;

  for (auto ch : input) {
    if (IsUnicodeSpace(ch)) {
      if (!lastWasSpace) {
        result.push_back(ch);
      }
      lastWasSpace = true;
    } else {
      result.push_back(ch);
      lastWasSpace = false;
    }
  }

  return result;
}

KernelBpeTokenizer::KernelBpeTokenizer(const BpeModelConf& conf)
    : bpe_conf_(conf) {
  model_name_ = conf.name_ == nullptr ? "" : conf.name_;
  CreateUnicodeByteEncoder();
};

OrtStatusPtr KernelBpeTokenizer::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  // note: if the attribute doesn't exist in op node, GetOpAttribute doesn't return a failed status;
  std::string vocab;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "vocab", vocab));
  if (vocab.empty()) {
    return OrtW::CreateStatus("vocabulary shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  std::string merges;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "merges", merges));
  if (merges.empty()) {
    return OrtW::CreateStatus("merges shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "padding_length", padding_length_));
  if (padding_length_ != -1 && padding_length_ <= 0) {
    return OrtW::CreateStatus("padding_length should be more than 0 or equal -1", ORT_INVALID_ARGUMENT);
  }

  std::string model_name;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "model_name", model_name));
  if (!model_name.empty()) {
    model_name_ = model_name;
  }

  std::stringstream vocab_stream(vocab);
  std::stringstream merges_stream(merges);
  bbpe_tokenizer_ = std::make_unique<BpeModel>();
  auto status = bbpe_tokenizer_->Load(vocab_stream,
                                      merges_stream,
                                      bpe_conf_.get().unk_token_,
                                      bpe_conf_.get().GetSpecialTokens().c_str(),
                                      bpe_conf_.get().spm_model_);
  if (!status.IsOk()) {
    return (OrtStatusPtr)status;
  }

  std::string added_token;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "added_token", added_token));
  status = bbpe_tokenizer_->LoadAddedTokens(added_token.c_str());
  if (!status.IsOk()) {
    return (OrtStatusPtr)status;
  }

  // TODO: need to check if the special token ids are the same as the ones in HFTokenizer
  if (bpe_conf_.get().bos_token_ != nullptr) {
    bos_token_id_ = bbpe_tokenizer_->GetTokenId(bpe_conf_.get().bos_token_);
  }
  if (bpe_conf_.get().eos_token_ != nullptr) {
    eos_token_id_ = bbpe_tokenizer_->GetTokenId(bpe_conf_.get().eos_token_);
  }
  if (bpe_conf_.get().pad_token_ != nullptr) {
    pad_token_id_ = bbpe_tokenizer_->GetTokenId(bpe_conf_.get().pad_token_);
  }

  return {};
}

uint32_t KernelBpeTokenizer::GetTokenId(const std::string& token) const {
  auto id = bbpe_tokenizer_->GetAddedTokenId(token);
  if (id != bpe::kInvalidTokenId) {
    return id;
  }

  return bbpe_tokenizer_->GetTokenId(token);
}

/*
Read more here: https://github.com/huggingface/transformers/blob/60bb571e993b7d73257fb64044726b569fef9403/src/transformers/convert_slow_tokenizer.py#L1454

Note: this is similar to the BPE CreateByteEncoder, however for decoding the .tiktoken bytes
we need to store the strings rather than their IDs, and thereby need a separate map.
*/
void KernelBpeTokenizer::CreateUnicodeByteEncoder() {
  char32_t index = 256;
  for (char32_t i = 0; i < 256; ++i) {
    if ((i >= 0 && i < 33) || (i >= 127 && i < 161) || (i == 173)) {
      unicode_byte_encoder_[i] = ustring::EncodeUTF8Char(index++);
    } else {
      unicode_byte_encoder_[i] = ustring::EncodeUTF8Char(i);
    }
  }
}

std::vector<int64_t> KernelBpeTokenizer::Tokenize(ustring& input,
                                                  int64_t max_length,
                                                  bool compute_offset_mapping,
                                                  std::list<OffsetMappingType>& offset_map) const {
  std::vector<int64_t> res;

  bool clean_up_spaces = false;
  if (ModelName() == kModel_CLIP) {
    clean_up_spaces = true;
    // Merges consecutive '\s+' for CLIP
    /*
      text = re.sub(r"\s+", " ", text)
      text = text.strip()
    */
    ustring str = RemoveConsecutiveSpaces(input);
    if (!str.empty() && IsUnicodeSpace(str.front())) {
      str.erase(str.begin());
    }
    if (!str.empty() && IsUnicodeSpace(str.back())) {
      str.pop_back();
    }
    // remove newlines as CLIP ignores them (treats them as whitespace which is then cleaned)
    str.erase(std::remove(str.begin(), str.end(), U'\n'), str.end());
    str.erase(std::remove(str.begin(), str.end(), U'\r'), str.end());
    input = str;
  }

  if (AllSpaceUstring(input) && ModelName() == kModel_CLIP) {
    // Add BOS and EOS token to result
    res.push_back(bos_token_id_);
    res.push_back(eos_token_id_);
    return res;
  }

  bool add_bos_token = false;
  if (add_bos_token_.has_value()) {
    add_bos_token = add_bos_token_.value();
  } else if (IsBosEosRequired(ModelName())) {
    add_bos_token = true;
  }
  if (add_bos_token) {
    res.push_back(bos_token_id_);
  }

  bool add_eos_token = false;
  if (add_eos_token_.has_value()) {
    add_eos_token = add_eos_token_.value();
  } else if (IsBosEosRequired(ModelName())) {
    add_eos_token = true;
  }

  if (ModelName() == kModel_CLIP) {
    // Convert to lowercase
    std::transform(input.begin(), input.end(), input.begin(), [](char32_t c) { return static_cast<char32_t>(ToLower(c)); });
    // for (auto& c : input) {
    //   if (c < 256) {
    //     c = _Py_ctype_tolower[c];
    //   }
    // }
  }

  // Parse input
  auto special_token_split_res = bbpe_tokenizer_->SplitByAddedAndSpecial(input);
  bpe::TokenWithRegularExp regcmp;

  for (auto& seg_id : special_token_split_res) {
    if (static_cast<int64_t>(res.size()) >= max_length) break;

    if (seg_id.second != bpe::kInvalidTokenId) {
      res.push_back(seg_id.second);
      continue;
    }

    // Note: keep ptr to make sure the string_view is valid in the following process
    std::u32string str(seg_id.first);
    regcmp.Set(str.c_str());

    size_t offset = 0;
    OffsetMappingType offset_mapping;

    if (compute_offset_mapping) {
      if (add_bos_token) {
        // Add offset mapping for BOS token
        offset_mapping.push_back(std::make_pair(0, 0));
      }
    }

    while (static_cast<int64_t>(res.size()) < max_length) {
      std::string regex_expr = "";
      if (ModelName() == kModel_Llama){
        regex_expr = regcmp.LLAMA_REGEX_PATTERN_1;
      } else {
        // default to GPT2 regex
        regex_expr = regcmp.GPT2_REGEX_PATTERN;
      }
      auto [b, tok] = regcmp.GetNextToken(regex_expr);

      if (!b) break;

      std::string utf8_token = std::string(ustring(tok));

      size_t space_dif = 0;
      if (compute_offset_mapping) {
        // Handle special case for offset mapping
        if (utf8_token.at(0) == ' ') {
          offset++;
          space_dif = -1;  // account for spaces used in offset map algorithm in bpe(byte_list_)
        }
      }

      std::list<std::pair<uint32_t, uint32_t>> byte_list;
      std::string token_bytes;
      token_bytes.reserve(utf8_token.size() * 2);
      size_t token_len = utf8_token.length();
      size_t end_diff = 0;
      if (clean_up_spaces) {
        // Whitespace clean
        utf8_token.erase(std::remove(utf8_token.begin(), utf8_token.end(), U' '), utf8_token.end());
        token_len = utf8_token.length() - 1;
      }

      for (size_t i = 0; i < token_len; i++) {
        token_bytes += unicode_byte_encoder_[static_cast<unsigned char>(utf8_token[i])];
      }

      if (clean_up_spaces) {
        end_diff = token_bytes.length();
        if (!utf8_token.empty()) {
          token_bytes += unicode_byte_encoder_[static_cast<unsigned char>(utf8_token.back())];
          token_bytes += "</w>";
        }
        end_diff = token_bytes.length() - end_diff;
      }

      auto id = bbpe_tokenizer_->GetTokenId(token_bytes);
      if (id != bpe::kInvalidTokenId) {
        byte_list.push_back(std::make_pair(id, ort_extensions::narrow<uint32_t>(utf8_token.size())));
      } else {
        token_len = token_bytes.length();
        for (size_t i = 0; i < token_len - end_diff; /* i++ */) {
          size_t j = ustring::UTF8Len(token_bytes[i]);
          byte_list.push_back(std::make_pair(bbpe_tokenizer_->GetTokenId(token_bytes.substr(i, j)), ort_extensions::narrow<uint32_t>(j)));
          i += j;
        }
        if (end_diff > 0) {
          byte_list.push_back(std::make_pair(
            bbpe_tokenizer_->GetTokenId(token_bytes.substr(token_len - end_diff, end_diff)), ort_extensions::narrow<uint32_t>(end_diff)));
        }
      }

      // Perform BPE
      bbpe_tokenizer_->PerformBPE(byte_list);

      // Add output to result
      for (auto p : byte_list) {
        if (static_cast<int64_t>(res.size()) >= max_length) {
          break;
        }

        res.push_back(p.first);

        if (compute_offset_mapping) {
          if (clean_up_spaces) {
            offset_mapping.emplace_back(std::make_pair(offset, ort_extensions::narrow<size_t>(offset + p.second)));
            offset += p.second;
          } else {
            offset_mapping.emplace_back(std::make_pair(
                offset,
                ort_extensions::narrow<size_t>(offset + (size_t)p.second + space_dif)));
            offset += ((size_t)p.second + space_dif);
          }
        }
      }
    }

    if (compute_offset_mapping) {
      if (add_eos_token) {
        // Add offset mapping for EOS token
        offset_mapping.emplace_back(std::make_pair(0, 0));
      }
      // Add offset mappings for input in this instance to list of offset mappings for all inputs
      offset_map.emplace_back(offset_mapping);
    }
  }

  if (add_eos_token) {
    // Add EOS token to result
    res.push_back(eos_token_id_);
  }

  return res;
}

std::vector<int64_t> KernelBpeTokenizer::SpmTokenize(ustring& input,
                                                     int64_t max_length_i64,
                                                     bool compute_offset_mapping,
                                                     std::list<OffsetMappingType>& offset_map) const {
  std::vector<int64_t> res;

  // Add BOS token to result
  res.push_back(bos_token_id_);

  size_t max_length = static_cast<size_t>(max_length_i64);
  // Parse input
  bool add_dummy_prefix = false;
  if (ModelName() == kModel_Llama) {
    add_dummy_prefix = true;
  }
  auto special_token_split_res = bbpe_tokenizer_->SplitByAddedAndSpecial(input);
  for (auto& seg_id : special_token_split_res) {
    if (res.size() >= max_length) break;

    if (seg_id.second != bpe::kInvalidTokenId) {
      res.push_back(seg_id.second);
      continue;
    }

    // Note: keep ptr to make sure the string_view is valid in the following process
    std::u32string ustr(seg_id.first);
    if (add_dummy_prefix) {
      ustr.insert(ustr.begin(), 0x2581);  // UTF-8 string '\xe2\x96\x81'
      add_dummy_prefix = false;           // only add dummy prefix once
    }

    size_t offset = 0;
    size_t char_pos = 0;
    OffsetMappingType offset_mapping;

    if (compute_offset_mapping) {
      offset_mapping.push_back(std::make_pair(0, 0));
    }

    // Get byte encodings prior to performing BPE
    std::list<std::pair<uint32_t, uint32_t>> byte_list;

    while (res.size() < max_length && char_pos <= ustr.length()) {
      bool split_now = false;
      if (char_pos == ustr.length()) {
        split_now = true;
      }

      // temporary split logic, will be replaced regex based split after it is implemented
      if (!split_now && byte_list.size() > 10) {
        auto is_split_char = [](char32_t ch) {
          return ch == U' ' || ch == U'\n' || ch == U'\r' || ch == U'▁';
        };
        if (!is_split_char(ustr[char_pos - 1]) && is_split_char(ustr[char_pos])) {
            split_now = true;
        }
        // split immediately to avoid too long byte_list for extreme cases, which is slow.
        if (!split_now && byte_list.size() > 100) {
          split_now = true;
        }
      }

      if (split_now) {
        // Perform BPE
        bbpe_tokenizer_->PerformBPE(byte_list);

        // Add output to result
        for (auto p : byte_list) {
          if (res.size() >= max_length) {
            break;
          }

          res.push_back(p.first);

          if (compute_offset_mapping) {
            offset_mapping.emplace_back(std::make_pair(
                offset,
                ort_extensions::narrow<size_t>(offset + (size_t)p.second)));
            offset += ((size_t)p.second);
          }
        }

        byte_list.clear();
      }

      if (char_pos == ustr.length()) {
        break;
      }

      auto chr = ustr[char_pos];
      if (chr == U' ') {
        chr = 0x2581;  // UTF-8 string '\xe2\x96\x81'
      }

      std::u32string token_ucs = {chr};
      std::string token_s = (std::string)ustring(token_ucs);
      auto id = bbpe_tokenizer_->GetTokenId(token_s);
      if (id == bpe::kInvalidTokenId) {
        for (auto chr : token_s) {
          auto byte_id = bbpe_tokenizer_->GetTokenId({chr});
          byte_list.emplace_back(byte_id, 1);
        }
      } else {
        byte_list.emplace_back(id, ort_extensions::narrow<uint32_t>(token_s.length()));
      }

      char_pos++;
    }

    if (compute_offset_mapping) {
      // Add offset mappings for input in this instance to list of offset mappings for all inputs
      offset_map.emplace_back(offset_mapping);
    }
  }

  if (res.size() > 0 && res.front() == bos_token_id_) {
    if (add_bos_token_.has_value() && add_bos_token_.value() == false) {
      res.erase(res.begin());
    }
  }

  return res;
}

OrtxStatus KernelBpeTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                       ortc::Tensor<int64_t>& tokenize_output,
                                       std::optional<ortc::Tensor<int64_t>*> attention_mask,
                                       std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
  // Setup inputs
  std::vector<std::string> str_input{input.Data()};
  std::list<OffsetMappingType> offset_map;
  const auto& input_dim = input.Shape();

  std::vector<std::vector<int64_t>> tokenize_results;

  // Only compute offset mapping if optional output for it exists.
  bool compute_offset_mapping = false;
  if (offset_mapping.has_value()) {
    compute_offset_mapping = true;
  }

  auto tok_fun = &KernelBpeTokenizer::Tokenize;
  if (bpe_conf_.get().spm_model_) {
    tok_fun = &KernelBpeTokenizer::SpmTokenize;
  }

  for (auto& str : str_input) {
    ustring ustr = ustring(str);
    tokenize_results.emplace_back(
        (this->*tok_fun)(
            ustr,
            padding_length_ < 0 ? (std::numeric_limits<uint32_t>::max)() : padding_length_,
            compute_offset_mapping,
            offset_map));
  }

  size_t max_length = 0;
  if (padding_length_ == -1) {
    for (auto& res : tokenize_results) {
      max_length = (std::max)(max_length, res.size());
    }
  } else {
    max_length = static_cast<size_t>(padding_length_);
  }

  std::vector<int64_t> output_dim = input_dim;
  output_dim.push_back(max_length);

  std::vector<int64_t> offset_dim = output_dim;
  offset_dim.push_back(2);  // tuple of offsets for each input id

  auto* token = tokenize_output.Allocate(output_dim);
  if (attention_mask.has_value()) {
    auto* mask = (*attention_mask)->Allocate(output_dim);
    int idx = 0;
    for (auto& res : tokenize_results) {
      for (int64_t id : res) {
        mask[idx] = 1;
        idx++;
      }

      for (size_t i = res.size(); i < max_length; i++) {
        mask[idx] = 0;
        idx++;
      }
    }
  }
  if (offset_mapping.has_value()) {
    auto* offset = (*offset_mapping)->Allocate(offset_dim);
    int idx2 = 0;
    for (auto& res : offset_map) {
      for (auto& mapping : res) {
        offset[idx2] = mapping.first;
        idx2++;
        offset[idx2] = mapping.second;
        idx2++;
      }
    }
  }
  int idx = 0;
  for (auto& res : tokenize_results) {
    for (int64_t id : res) {
      token[idx] = id;
      idx++;
    }

    for (size_t i = res.size(); i < max_length; i++) {
      token[idx] = pad_token_id_;
      idx++;
    }
  }

  return {};
}

static const auto kGPT2Configuration = BpeModelConf();
GPT2Tokenizer::GPT2Tokenizer()
    : KernelBpeTokenizer(kGPT2Configuration) {}

static const auto kRobertaConfiguration = BpeModelConf{
    kModel_Roberta,  // name
    "<unk>",         // unk_token
    "<s>",           // bos_token
    "</s>",          // eos_token
    "<pad>"};        // pad_token

RobertaTokenizer::RobertaTokenizer()
    : KernelBpeTokenizer(kRobertaConfiguration) {}

static const auto kCLIPConfiguration = BpeModelConf{
    kModel_CLIP,        // name
    "<|endoftext|>",    // unk_token
    "<|startoftext|>",  // bos_token
    "<|endoftext|>",    // eos_token
    "<|endoftext|>"};   // pad_token

CLIPTokenizer::CLIPTokenizer()
    : KernelBpeTokenizer(kCLIPConfiguration) {}

static const auto kSpmConfiguration = BpeModelConf{
    kModel_Llama,  // name
    "<unk>",       // unk_token
    "<s>",         // bos_token
    "</s>",        // eos_token
    "",            // pad_token
    true};

SpmTokenizer::SpmTokenizer()
    : KernelBpeTokenizer(kSpmConfiguration) {}

JsonFastTokenizer::JsonFastTokenizer() : KernelBpeTokenizer(kGPT2Configuration) {}

std::string JsonFastTokenizer::TokenBytesToString(std::vector<uint8_t>& bytes) {
    std::string result;
    for (auto c : bytes) {
        result += unicode_byte_encoder_[static_cast<unsigned char>(c)];
    }
    return result;
}

// Custom hash function for the vector key
struct VectorHash {
    size_t operator()(const std::vector<uint8_t>& v) const {
        std::hash<uint8_t> hasher;
        size_t seed = 0;
        for (uint8_t i : v) {
            seed ^= hasher(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Custom equality function for the vector key
struct VectorEqual {
    bool operator()(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) const {
        return a == b;
    }
};

OrtxStatus JsonFastTokenizer::LoadAddedTokens(const json& tok_json, const ort_extensions::bpe::TokenJsonConfig& config) {
  auto added_tokens = tok_json.find("added_tokens");
  if (added_tokens != tok_json.end()) {
    for (const auto& token : *added_tokens) {
      bpe::AddedToken added_token;
      added_token.id_ = token.value("id", 0);
      added_token.token_type_ = token.value("__type", "");
      added_token.content_ = token.value("content", "");
      added_token.lstrip_ = token.value("lstrip", false);
      added_token.normalized_ = token.value("normalized", false);
      added_token.rstrip_ = token.value("rstrip", false);
      added_token.single_word_ = token.value("single_word", false);
      added_token.special_ = token.value("special", false);

      added_tokens_.emplace_back(added_token);
      if (added_token.content_ == config.bos_token_) {
        bos_token_id_ = added_token.id_;
      } else if (added_token.content_ == config.eos_token_) {
        eos_token_id_ = added_token.id_;
      } else if (added_token.content_ == config.pad_token_) {
        pad_token_id_ = added_token.id_;
      }
    }
  }

  return bbpe_tokenizer_->LoadAddedTokens(added_tokens_);
}

// Helper methods (to be added to the class declaration)
bool JsonFastTokenizer::CheckForSpmModel(const json& tok_json) {
  auto decoder_node = tok_json.find("decoder");
  if (decoder_node != tok_json.end()) {
    auto decoders_node = decoder_node->find("decoders");
    if (decoders_node != decoder_node->end() && decoders_node->is_array()) {
      for (const auto& step : *decoders_node) {
        std::string type = step.value("type", "");
        if (type == "Replace") {
          std::string target = step.value("/pattern/String"_json_pointer, "");
          if (target == spm_escaped_space) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

void JsonFastTokenizer::UpdateTokenAdditionFlags(const json& tok_json, const ort_extensions::bpe::TokenJsonConfig& config) {
  if (!config.add_bos_token_ && !config.bos_token_.empty()) {
    auto post_processor = tok_json.find("post_processor");
    if (post_processor != tok_json.end()) {
      std::string text = post_processor->dump();
      if (text.find(config.bos_token_) != std::string::npos) {
        add_bos_token_ = true;
      }
      if (text.find(config.eos_token_) != std::string::npos) {
        add_eos_token_ = true;
      }
    }
  }
}

OrtxStatus JsonFastTokenizer::Load(const ort_extensions::bpe::TokenJsonConfig& config) {
  std::string voc_file = config.GetVocabDataFile();
  std::ifstream ifs = path(voc_file).open();
  if (!ifs.is_open()) {
    return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open json file: " + voc_file);
  }

  nlohmann::json tok_json;
  ifs >> tok_json;

  const char token_sub[] = "Tokenizer";
  model_name_ = config.tokenizer_class_.substr(0, config.tokenizer_class_.find(token_sub));
  json_conf_.name_ = model_name_.c_str();
  json_conf_.bos_token_ = config.bos_token_.c_str();
  json_conf_.eos_token_ = config.eos_token_.c_str();
  json_conf_.unk_token_ = config.unk_token_.c_str();
  json_conf_.pad_token_ = config.pad_token_.c_str();

  // re-bind the configuration object
  bpe_conf_ = json_conf_;

  // Check for SPM model
  json_conf_.spm_model_ = CheckForSpmModel(tok_json);

  auto model_node = tok_json.find("model");
  if (model_node == tok_json.end()) {
    return OrtxStatus(kOrtxErrorCorruptData, "Failed to get model node from tokenizer.json");
  }

  bbpe_tokenizer_ = std::make_unique<BpeModel>();
  OrtxStatus status = bbpe_tokenizer_->Load(*model_node,
                                            bpe_conf_.get().GetSpecialTokens().c_str(),
                                            bpe_conf_.get().spm_model_);
  if (!status.IsOk()) {
    return status;
  }

  status = LoadAddedTokens(tok_json, config);
  if (!status.IsOk()) {
    return status;
  }

  add_bos_token_ = config.add_bos_token_;
  add_eos_token_ = config.add_eos_token_;
  UpdateTokenAdditionFlags(tok_json, config);

  return status;
}

OrtxStatus JsonFastTokenizer::LoadTikTokenBase64(const ort_extensions::bpe::TokenJsonConfig& config) {
  std::string voc_file = config.GetVocabDataFile();
  std::ifstream ifs = path(voc_file).open();
  if (!ifs.is_open()) {
    return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open json file: " + voc_file);
  }

  std::unordered_map<std::string, uint32_t> vocab;
  std::vector<std::pair<std::string, std::string>> merges;
  std::unordered_map<std::vector<uint8_t>, uint32_t, VectorHash, VectorEqual> bpe_ranks;

  std::string line;
  while (std::getline(ifs, line)) {
    if (!line.empty()) {
      std::istringstream lineStream(line);
      std::string token;
      uint32_t rank;
      while (lineStream >> token >> rank) {
          // Decode base64 token and convert rank to int
          std::vector<uint8_t> decoded_token;
          base64_decode(token, decoded_token);
          // Store bpe token and rank
          bpe_ranks[decoded_token] = rank;
      }
    }
  }

  std::vector<std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>> byte_merges;

  bbpe_tokenizer_ = std::make_unique<BpeModel>();
  
  for (const auto& item : bpe_ranks) {
    std::vector<uint8_t> token = item.first;
    uint32_t rank = item.second;
    vocab[JsonFastTokenizer::TokenBytesToString(token)] = rank;
    
    if (token.size() == 1) {
        continue;
    }
    
    std::vector<std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>> local;
    for (size_t index = 1; index < token.size(); index++) {
        std::vector<uint8_t> piece_l(token.begin(), token.begin() + index);
        std::vector<uint8_t> piece_r(token.begin() + index, token.end());
        if (bpe_ranks.count(piece_l) && bpe_ranks.count(piece_r)) {
            local.emplace_back(piece_l, piece_r, rank);
        }
    }
    
    auto compare_bpe_tuples = [&](const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>& a,
                                      const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>& b) {
      // Compare comparator based on the ranks in bpe_ranks
      return bpe_ranks[std::get<0>(a)] < bpe_ranks[std::get<0>(b)] ||
              (bpe_ranks[std::get<0>(a)] == bpe_ranks[std::get<0>(b)] && bpe_ranks[std::get<1>(a)] < bpe_ranks[std::get<1>(b)]);
    };

    std::sort(local.begin(), local.end(), compare_bpe_tuples);
    
    byte_merges.insert(byte_merges.end(), local.begin(), local.end());
  }

  // Custom comparator that compares the third element of the tuples
  auto compare_merge_tuples = [&](const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>& a,
                                      const std::tuple<std::vector<uint8_t>, std::vector<uint8_t>, uint32_t>& b) {
    return std::get<2>(a) < std::get<2>(b);
  };
  
  std::sort(byte_merges.begin(), byte_merges.end(), compare_merge_tuples);

  // Populate merges
  for (auto& val : byte_merges) {
    merges.push_back({JsonFastTokenizer::TokenBytesToString(std::get<0>(val)), JsonFastTokenizer::TokenBytesToString(std::get<1>(val))});
  }

  const char token_sub[] = "Tokenizer";
  model_name_ = config.tokenizer_class_.substr(0, config.tokenizer_class_.find(token_sub));
  json_conf_.name_ = model_name_.c_str();
  json_conf_.bos_token_ = config.bos_token_.c_str();
  json_conf_.eos_token_ = config.eos_token_.c_str();
  json_conf_.unk_token_ = config.unk_token_.c_str();
  json_conf_.pad_token_ = config.pad_token_.c_str();

  // re-bind the configuration object
  bpe_conf_ = json_conf_;

  OrtxStatus status = bbpe_tokenizer_->Load(vocab,
                                            merges,
                                            bpe_conf_.get().GetSpecialTokens().c_str(),
                                            false);

  if (!status.IsOk()) {
    return status;
  }

  std::string module_file = config.GetTikTokenModuleFile();
  std::ifstream module_ifs = path(module_file).open();
  if (!module_ifs.is_open()) {
    return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open module file: " + module_file);
  }

  nlohmann::json tok_json;
  module_ifs >> tok_json;
  status = LoadAddedTokens(tok_json, config);
  if (!status.IsOk()) {
    return status;
  }

  add_bos_token_ = config.add_bos_token_;
  add_eos_token_ = config.add_eos_token_;
  UpdateTokenAdditionFlags(tok_json, config);

  return status;
}

OrtxStatus JsonFastTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                      ortc::Tensor<int64_t>& tokenize_output,
                                      std::optional<ortc::Tensor<int64_t>*> attention_mask,
                                      std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
  return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
}

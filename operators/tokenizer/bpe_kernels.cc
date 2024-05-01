// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ortx_common.h"
#include "bpe_kernels.h"
#include "bpe_json.hpp"
#include "bpe_tokenizer.hpp"

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

static bool IsSpmModel(const std::string& model_name) {
  return model_name == kModel_Llama ||
         model_name == kModel_Gemma;
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
                                      IsSpmModel(ModelName()));
  if (!status.IsOk()) {
    return status.CreateOrtStatus();
  }

  std::string added_token;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "added_token", added_token));
  status = bbpe_tokenizer_->LoadAddedTokens(added_token.c_str());
  if (!status.IsOk()) {
    return status.CreateOrtStatus();
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

std::vector<int64_t> KernelBpeTokenizer::Tokenize(ustring& input,
                                                  int64_t max_length,
                                                  bool compute_offset_mapping,
                                                  std::list<OffsetMappingType>& offset_map) const {
  std::vector<int64_t> res;
  std::list<std::pair<uint32_t, uint32_t>> byte_list;

  bool clean_up_spaces = false;
  if (ModelName() == kModel_CLIP) {
    clean_up_spaces = true;
    // Merges consecutive '\s+' for CLIP
    /*
      text = re.sub(r"\s+", " ", text)
      text = text.strip()
    */
    ustring str = RemoveConsecutiveSpaces(input);
    if (IsUnicodeSpace(str.front())) {
      str.erase(str.begin());
    }
    if (IsUnicodeSpace(str.back())) {
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
      auto [b, tok] = regcmp.GetNextToken();

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

      // Get byte encodings prior to performing BPE
      byte_list.clear();

      if (clean_up_spaces) {
        // Whitespace clean
        utf8_token.erase(std::remove(utf8_token.begin(), utf8_token.end(), U' '), utf8_token.end());

        for (int i = 0; i < utf8_token.length(); i++) {
          if (i == utf8_token.length() - 1) {
            std::string boundary(1, utf8_token[i]);
            byte_list.push_back(std::make_pair(bbpe_tokenizer_->GetTokenId(boundary + "</w>"), 1));
          } else {
            byte_list.push_back(std::make_pair(bbpe_tokenizer_->ByteEncoder()[static_cast<unsigned char>(utf8_token[i])], 1));
          }
        }
      } else {
        for (char& cp : utf8_token) {
          byte_list.push_back(std::make_pair(bbpe_tokenizer_->ByteEncoder()[static_cast<unsigned char>(cp)], 1));
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
  std::list<std::pair<uint32_t, uint32_t>> byte_list;

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
    byte_list.clear();

    while (res.size() < max_length && char_pos < ustr.length()) {
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
    {
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
    }

    if (compute_offset_mapping) {
      // Add offset mappings for input in this instance to list of offset mappings for all inputs
      offset_map.emplace_back(offset_mapping);
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
  if (IsSpmModel(ModelName())) {
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
    ""};           // pad_token

SpmTokenizer::SpmTokenizer()
    : KernelBpeTokenizer(kSpmConfiguration) {}

JsonFastTokenizer::JsonFastTokenizer() : KernelBpeTokenizer(kGPT2Configuration) {}

OrtxStatus JsonFastTokenizer::Load(const ort_extensions::bpe::TokenJsonConfig& config) {
  std::string voc_file = config.GetVocabDataFile();
  std::ifstream ifs(voc_file);
  if (!ifs.is_open()) {
    return OrtxStatus(kOrtxErrorInvalidFile, "Failed to open json file: " + voc_file);
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

  // consider to use SAX parser for large json file
  nlohmann::json tok_json;
  ifs >> tok_json;
  auto model_node = tok_json.find("model");
  if (model_node == tok_json.end()) {
    return OrtxStatus(kOrtxErrorCorruptData, "Failed to get model node from tokenizer.json");
  }

  bbpe_tokenizer_ = std::make_unique<BpeModel>();
  auto status = bbpe_tokenizer_->Load(*model_node,
                                      bpe_conf_.get().GetSpecialTokens().c_str(),
                                      IsSpmModel(ModelName()));

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

  if (!status.IsOk()) {
    return status;
  }

  status = bbpe_tokenizer_->LoadAddedTokens(added_tokens_);
  if (!status.IsOk()) {
    return status;
  }

  add_bos_token_ = config.add_bos_token_;
  add_eos_token_ = config.add_eos_token_;
  // add_bos_token is default as false, we need to check post_processor json to see if it is true
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

  return status;
}

OrtxStatus JsonFastTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                      ortc::Tensor<int64_t>& tokenize_output,
                                      std::optional<ortc::Tensor<int64_t>*> attention_mask,
                                      std::optional<ortc::Tensor<int64_t>*> offset_mapping) const {
  return KernelBpeTokenizer::Compute(input, tokenize_output, attention_mask, offset_mapping);
}

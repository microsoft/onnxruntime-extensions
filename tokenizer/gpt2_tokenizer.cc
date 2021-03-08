// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Partial code comes from other Microsoft employee.

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <list>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <functional>
#include <codecvt>
#include <mutex>

#include "gpt2_tokenizer.hpp"
#include "nlohmann/json.hpp"
#include "kernels/kernels.h"
#include "kernels/string_common.h"
#include "unicode.h"

class SpecialTokenMap {
 public:
  void Add(ustring p_str, int p_id) {
    auto it = token_map_.find(p_str);
    if (it != token_map_.end()) {
      if (it->second != p_id) {
        throw std::runtime_error("Duplicate special tokens");
      }
    } else {
      token_map_[p_str] = p_id;
      token_list_.push_back(SpecialTokenInfo(std::move(p_str), p_id));
    }
  }

  std::list<std::pair<ustring, int>> SplitBySpeicalTokens(ustring input) const {
    std::list<std::pair<ustring, int>> res;
    res.emplace_back(std::move(input), -1);
    for (const auto& st : token_list_) {
      std::list<std::pair<ustring, int>> new_split_res;
      for (auto& str : res) {
        if (str.second != -1) {
          new_split_res.push_back(std::move(str));
          continue;
        }
        auto it = str.first.begin();
        size_t search_pos = 0;
        while (it != str.first.end()) {
#if defined(__APPLE__)
          auto search_it = std::search(it, str.first.end(), st.str.begin(), st.str.end());
#else
          auto search_it = std::search(it, str.first.end(),
                                       std::boyer_moore_searcher(st.str.begin(), st.str.end()));
#endif
          if (search_it == str.first.end()) {
            new_split_res.emplace_back(str.first.substr(search_pos), -1);
            break;
          }
          auto prefixLen = search_it - it;
          if (prefixLen != 0) {
            new_split_res.emplace_back(str.first.substr(search_pos, prefixLen), -1);
            search_pos += prefixLen;
          }
          new_split_res.emplace_back(str.first.substr(search_pos, st.str.size()), st.id);
          it = search_it + st.str.size();
          search_pos += st.str.size();
        }
      }
      std::swap(new_split_res, res);
    }
    return res;
  }

 private:
  struct SpecialTokenInfo {
    ustring str;
    int id;

    SpecialTokenInfo(ustring p_str, int p_id)
        : str(std::move(p_str)), id(p_id) {
      if (str.empty()) {
        throw std::runtime_error("Empty special token.");
      }
    }
  };

  std::list<SpecialTokenInfo> token_list_;
  std::unordered_map<ustring, int> token_map_;
};

using json = nlohmann::json;
class VocabData {
 public:
  VocabData()
      : unk_id_(-1) {
  }

  struct BpeNode {
    int id;
    int value;
  };

  void Load(std::istream& vocab_stream, std::istream& merges_stream, const char* unk_token, const char* special_tokens) {
    json tok_json;
    vocab_stream >> tok_json;
    vocab_map_ = std::move(tok_json.get<std::unordered_map<std::string, int>>());

    auto it = vocab_map_.find(unk_token);
    if (it != vocab_map_.end()) {
      unk_id_ = it->second;
    } else {
      int id = static_cast<int>(vocab_map_.size());
      vocab_map_[unk_token] = id;
      std::cerr << "Special token (" << unk_token << ") have been added in the vocabulary." << std::endl;
    }

    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> str_convert;
    for (auto i = 33; i <= 126; ++i) {
      byte_encoder_[i] = GetVocabIndex(str_convert.to_bytes((char32_t)i));
    }
    for (auto i = 161; i <= 172; ++i) {
      byte_encoder_[i] = GetVocabIndex(str_convert.to_bytes((char32_t)i));
    }
    for (auto i = 174; i <= 255; ++i) {
      byte_encoder_[i] = GetVocabIndex(str_convert.to_bytes((char32_t)i));
    }

    int index = 256;
    for (auto i = 0; i < 33; ++i) {
      byte_encoder_[i] = GetVocabIndex(str_convert.to_bytes((char32_t)(index++)));
    }
    for (auto i = 127; i < 161; ++i) {
      byte_encoder_[i] = GetVocabIndex(str_convert.to_bytes((char32_t)(index++)));
    }
    byte_encoder_[173] = GetVocabIndex(str_convert.to_bytes((char32_t)(index++)));

    index = 0;
    std::string line;
    while (std::getline(merges_stream, line)) {
      line = std::regex_replace(line, std::regex("\r"), "");
      if (line.empty()) continue;
      if ((line[0] == '#') && (index == 0)) continue;
      auto pos = line.find(' ');
      if (pos == std::string::npos) {
        throw std::runtime_error("Cannot know how to parse line: " + line);
      }
      std::string w1 = line.substr(0, pos);
      std::string w2 = line.substr(pos + 1);
      int iw1 = GetVocabIndex(w1);
      int iw2 = GetVocabIndex(w2);
      int iww = GetVocabIndex(w1 + w2);
      std::pair<int, int> key{iw1, iw2};
      BpeNode value{iww, index++};
      bpe_map_[key] = value;
    }

    if (special_tokens != nullptr) {
      std::istringstream istrea(special_tokens);

      while (istrea >> line) {
        if (line.empty()) continue;
        line = std::regex_replace(line, std::regex("\r"), "");
        ustring line_32(line);
        int id = static_cast<int>(vocab_map_.size());
        if (auto it = vocab_map_.find(line); it != vocab_map_.end()) {
          id = it->second;
        } else {
          vocab_map_[line] = id;
        }
        special_tokens_.Add(std::move(line_32), id);
      }
    }

    id2token_map_.resize(vocab_map_.size());
    for (const auto& [t, i] : vocab_map_) {
      id2token_map_[i] = t;
    }
  }

  void bpe(std::list<int>& vals) const {
    while (vals.size() >= 2) {
      auto pos_it = vals.end();
      int minval = std::numeric_limits<int>::max();
      int ori_id1 = 0, ori_id2 = 0;
      int aim_id = 0;
      for (auto it = vals.begin(); it != vals.end(); ++it) {
        auto it2 = it;
        ++it2;
        if (it2 == vals.end()) break;
        auto map_it = bpe_map_.find({*it, *it2});
        if (map_it == bpe_map_.end()) continue;
        if (minval > map_it->second.value) {
          ori_id1 = *it;
          ori_id2 = *it2;
          minval = map_it->second.value;
          pos_it = it;
          aim_id = map_it->second.id;
        }
      }
      if (pos_it == vals.end()) break;

      pos_it = vals.erase(pos_it);
      *pos_it = aim_id;
      for (++pos_it; pos_it != vals.end(); ++pos_it) {
        if (*pos_it != ori_id1) continue;
        auto it2 = pos_it;
        ++it2;
        if (it2 == vals.end()) break;
        if (*it2 != ori_id2) continue;
        pos_it = vals.erase(pos_it);
        *pos_it = aim_id;
      }
    }
  }

  const auto& ByteEncoder() const {
    return byte_encoder_;
  }

  auto SplitBySpeicalTokens(const ustring& input) const {
    return special_tokens_.SplitBySpeicalTokens(input);
  }

  size_t VocabSize() const { return vocab_map_.size(); }

  int TokenToID(const std::string& input) const {
    auto it = vocab_map_.find(input);
    if (it == vocab_map_.end()) {
      throw std::runtime_error("Token not found: " + input);
    }
    return it->second;
  }

  const std::string& IdToToken(int id) const {
    if ((id < 0) || (id >= id2token_map_.size())) {
      throw std::runtime_error("Invalid ID: " + std::to_string(id));
    }
    return id2token_map_[id];
  }

 private:
  int GetVocabIndex(const std::string& str) {
    auto it = vocab_map_.find(str);
    if (it == vocab_map_.end()) {
      throw std::runtime_error("Cannot find word in vocabulary: " + str);
    }
    return it->second;
  }

 private:
  struct hash_pair {
    template <class T1, class T2>
    size_t operator()(const std::pair<T1, T2>& p) const {
      auto hash1 = std::hash<T1>{}(p.first);
      auto hash2 = std::hash<T2>{}(p.second);
      return hash1 ^ (hash2 << 16);
    }
  };
  std::unordered_map<std::pair<int, int>, BpeNode, hash_pair> bpe_map_;

  int byte_encoder_[256] = {};
  std::unordered_map<std::string, int> vocab_map_;
  std::vector<std::string> id2token_map_;

  int unk_id_;
  SpecialTokenMap special_tokens_;
};

class TokenWithRegularExp {
 public:
  void Set(std::u32string_view val) {
    m_text = val;
  }

  std::pair<bool, std::u32string_view> GetNextToken() {
    while (!m_text.empty()) {
      auto res = TryMatch();
      if (res.empty()) {
        m_text = m_text.substr(1);
        continue;
      }
      return {true, res};
    }
    return {false, {}};
  }

 private:
  std::u32string_view TryMatch() {
    // python pattern:
    // 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

    // 's|'t|'re|'ve|'m|'ll|'d|
    // Note: the sequencial of the following if should not be switched, which follows the python regex's syntax
    if ((m_text[0] == U'\'') && (m_text.size() > 1)) {
      if ((m_text[1] == U's') || (m_text[1] == U't') ||
          (m_text[1] == U'm') || (m_text[1] == U'd')) {
        std::u32string_view res = m_text.substr(0, 2);
        m_text = m_text.substr(2);
        return res;
      }

      if (m_text.size() > 2) {
        if (((m_text[1] == U'r') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'v') && (m_text[2] == U'e')) ||
            ((m_text[1] == U'l') && (m_text[2] == U'l'))) {
          std::u32string_view res = m_text.substr(0, 3);
          m_text = m_text.substr(3);
          return res;
        }
      }
    }

    // ?\p{L}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::L)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::L) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::L) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?\p{N}+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (ufal::unilib::unicode::category(m_text[1]) & ufal::unilib::unicode::N)) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (ufal::unilib::unicode::category(m_text[0]) & ufal::unilib::unicode::N) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if ((ufal::unilib::unicode::category(m_text[i]) & ufal::unilib::unicode::N) == 0)
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // ?[^\s\p{L}\p{N}]+
    if ((m_text[0] == U' ') && (m_text.size() > 1) && (NotLNZ(m_text[1]))) {
      size_t i = 2;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }
    if (NotLNZ(m_text[0])) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!NotLNZ(m_text[i]))
          break;
      }
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    // \s+(?!\S)|\s+
    if ((m_text.size() >= 1) && (IsZ(m_text[0]))) {
      size_t i = 1;
      for (; i < m_text.size(); ++i) {
        if (!IsZ(m_text[i])) break;
      }
      if ((i > 1) && (i != m_text.size()))  //\s+(?!\S)
      {
        i--;
        std::u32string_view res = m_text.substr(0, i);
        m_text = m_text.substr(i);
        return res;
      }
      // \s+
      std::u32string_view res = m_text.substr(0, i);
      m_text = m_text.substr(i);
      return res;
    }

    return std::u32string_view{};
  }

  static bool IsZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    return (category & ufal::unilib::unicode::Z) != 0;
  }

  static bool NotLNZ(char32_t ch) {
    auto category = ufal::unilib::unicode::category(ch);
    if (category & ufal::unilib::unicode::L) return false;
    if (category & ufal::unilib::unicode::N) return false;
    if (category & ufal::unilib::unicode::Z) return false;
    return true;
  }

 private:
  std::u32string_view m_text;
};

//Note: the following logic comes from CPython: unicodetype_db.h (_PyUnicode_IsWhitespace)
bool IsUnicodeSpace(char32_t ch) {
  switch (ch) {
    case 0x0009:
    case 0x000A:
    case 0x000B:
    case 0x000C:
    case 0x000D:
    case 0x001C:
    case 0x001D:
    case 0x001E:
    case 0x001F:
    case 0x0020:
    case 0x0085:
    case 0x00A0:
    case 0x1680:
    case 0x2000:
    case 0x2001:
    case 0x2002:
    case 0x2003:
    case 0x2004:
    case 0x2005:
    case 0x2006:
    case 0x2007:
    case 0x2008:
    case 0x2009:
    case 0x200A:
    case 0x2028:
    case 0x2029:
    case 0x202F:
    case 0x205F:
    case 0x3000:
      return true;
  }
  return false;
}

bool IsEmptyUString(const ustring& str) {
  return std::all_of(str.begin(), str.end(), [](char32_t ch) { return IsUnicodeSpace(ch); });
}

KernelBpeTokenizer::KernelBpeTokenizer(OrtApi api, const OrtKernelInfo* info)
    : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(info, "vocab");
  if (vocab.empty()) {
    throw std::runtime_error("vocabulary shouldn't be empty.");
  }

  std::string merges = ort_.KernelInfoGetAttribute<std::string>(info, "merges");
  if (merges.empty()) {
    throw std::runtime_error("merges shouldn't be empty.");
  }

  if (!TryToGetAttribute<int64_t>("padding_length", padding_length_)) {
    padding_length_ = -1;
  }

  if (padding_length_ != -1 && padding_length_ <= 0) {
    throw std::runtime_error("padding_length should be more than 0 or equal -1");
  }

  std::stringstream vocabu_stream(vocab);
  std::stringstream merges_stream(merges);
  bbpe_tokenizer_ = std::make_shared<VocabData>();
  bbpe_tokenizer_->Load(vocabu_stream, merges_stream, "<|endoftext|>", "<|endoftext|>");
}

std::vector<int64_t> KernelBpeTokenizer::Tokenize(const ustring& input, int64_t max_length) {
  std::vector<int64_t> res;

  if (IsEmptyUString(input)) {
    return res;
  }

  auto special_token_split_res = bbpe_tokenizer_->SplitBySpeicalTokens(input);
  TokenWithRegularExp regcmp;

  for (auto& seg_id : special_token_split_res) {
    if (res.size() >= max_length) break;

    if (seg_id.second != -1) {
      res.push_back(seg_id.second);
      continue;
    }

    auto cur_input = std::move(seg_id.first);
    // Note: keep ptr to make sure the string_view is valid in the following process
    const char32_t* ptr = cur_input.c_str();
    regcmp.Set(ptr);

    while (res.size() < max_length) {
      auto [b, tok] = regcmp.GetNextToken();
      if (!b) break;

      std::string utf8_token = std::string(ustring(tok));

      byte_list_.clear();
      for (char& cp : utf8_token) {
        byte_list_.push_back(bbpe_tokenizer_->ByteEncoder()[static_cast<unsigned char>(cp)]);
      }

      bbpe_tokenizer_->bpe(byte_list_);

      for (auto p : byte_list_) {
        if (res.size() >= max_length) {
          break;
        }

        res.push_back(p);
      }
    }
  }

  return std::move(res);
}

void KernelBpeTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> str_input;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);
  OrtTensorDimensions input_dim(ort_, input);

  std::vector<std::vector<int64_t>> tokenize_results;
  for (auto& str : str_input) {
    tokenize_results.emplace_back(Tokenize(ustring(str), padding_length_ < 0 ? INT64_MAX : padding_length_));
  }

  size_t max_length = 0;
  if (padding_length_ == -1) {
    for (auto& res : tokenize_results) {
      max_length = std::max(max_length, res.size());
    }
  } else {
    max_length = padding_length_;
  }

  OrtTensorDimensions output_dim = input_dim;
  output_dim.push_back(max_length);
  OrtValue* tokenize_output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
  OrtValue* attention_mask = ort_.KernelContext_GetOutput(context, 1, output_dim.data(), output_dim.size());
  auto* token = ort_.GetTensorMutableData<int64_t>(tokenize_output);
  auto* mask = ort_.GetTensorMutableData<int64_t>(attention_mask);

  int idx = 0;
  for (auto& res : tokenize_results) {
    for (int64_t id : res) {
      token[idx] = id;
      mask[idx] = 1;
      idx++;
    }

    for (int i = res.size(); i < max_length; i++) {
      token[idx] = 0;
      mask[idx] = 0;
      idx++;
    }
  }
}

void* CustomOpBpeTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBpeTokenizer(api, info);
}

const char* CustomOpBpeTokenizer::GetName() const {
  return "GPT2Tokenizer";
}

size_t CustomOpBpeTokenizer::GetInputTypeCount() const {
  return 1;
}

ONNXTensorElementDataType CustomOpBpeTokenizer::GetInputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}
size_t CustomOpBpeTokenizer::GetOutputTypeCount() const {
  return 2;
}

ONNXTensorElementDataType CustomOpBpeTokenizer::GetOutputType(size_t index) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

const OrtCustomOp** LoadTokenizerSchemaList() {
  // create the global objects here to let the ORT catch the expection if any
  static std::unique_ptr<CustomOpBpeTokenizer> p_CoBpeTokenizer;
  static const OrtCustomOp* c_CustomOpList[2] = {nullptr};  // {&c_CoBpeTokenizer, nullptr};
  static std::mutex mtx_loaded;
  std::lock_guard<std::mutex> lck(mtx_loaded);
  if (p_CoBpeTokenizer.get() == nullptr) {
    p_CoBpeTokenizer = std::make_unique<CustomOpBpeTokenizer>();
    c_CustomOpList[0] = p_CoBpeTokenizer.get();
  }

  return c_CustomOpList;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include "ocos.h"
#include "narrow.h"

#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <charconv>
#include <optional>

#include "unescape.h"
#include "trietree.hpp"

// This Trie Tree is C++ implementation of
// https://github.com/BlinkDL/ChatRWKV/blob/main/rwkv_pip_package/src/rwkv/rwkv_tokenizer.py
// Perf optimized by leveraging C++ features, but the algorithm is the same.
class RWKVTrieTree : public ort_extensions::TrieTree<char> {
 public:
  static constexpr int kMaxTokenLength_ = 128;

  RWKVTrieTree(char ch = 0) : TrieTree(ch) {}

  // keep the same function for source code understanding.
  void add(const std::string& key, int idx = 0,
           std::optional<int> value = std::optional<int>()) {
    Add(key, idx, value);
  }

  int find_longest(const std::string& key, size_t& idx) {
    return FindLongest(key, idx);
  }
};

class TrieTokenizer {
 private:
  std::map<int, std::string> idx2token;
  RWKVTrieTree root;

 public:
  TrieTokenizer(const std::string& text_tokens) {
    std::istringstream file(text_tokens);
    std::string line;

    while (std::getline(file, line)) {
      auto l_ws = line.find(' ');
      auto r_ws = line.rfind(' ');
      if (l_ws == std::string::npos || r_ws == std::string::npos || l_ws == r_ws) {
        ORTX_CXX_API_THROW(MakeString("[TrieTokenizer] vocab line: ", line), ORT_RUNTIME_EXCEPTION);
      }

      int idx = 0;
      std::from_chars(line.data(), line.data() + line.size(), idx);
      if (idx == 0) {
        ORTX_CXX_API_THROW(MakeString("[TrieTokenizer] bad index in vocab line: ", line), ORT_RUNTIME_EXCEPTION);
      }

      std::string raw = line.substr(line.find(' ') + 1, line.rfind(' ') - line.find(' ') - 1);
      std::string x;
      int key_length = 0;
      if (ort_extensions::UnquoteString(raw, x)) {
        std::from_chars(line.data() + r_ws + 1, line.data() + line.size(), key_length);
      }
      if (x.length() != key_length) {
        ORTX_CXX_API_THROW(MakeString("[TrieTokenizer] bad len in vocab line: ", line), ORT_RUNTIME_EXCEPTION);
      }

      idx2token[idx] = x;
    }

    for (const auto& kv : idx2token) {
      root.add(kv.second, 0, kv.first);
    }
  }

  std::vector<int> encodeBytes(const std::string& src) {
    size_t idx = 0;
    std::vector<int> tokens;
    while (idx < src.length()) {
      auto result = root.find_longest(src, idx);
      tokens.push_back(result);
    }

    return tokens;
  }

  std::string decodeBytes(const std::vector<int>& tokens) {
    std::string result;
    for (const auto& i : tokens) {
      result += idx2token[i];
    }
    return result;
  }
};

struct KernelTrieTokenizer : public BaseKernel {
 private:
  std::shared_ptr<TrieTokenizer> tokenizer;

 public:
  KernelTrieTokenizer(const OrtApi& api, const OrtKernelInfo& info)
      : BaseKernel(api, info) {
    std::string text_tokens = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab");
    tokenizer = std::make_shared<TrieTokenizer>(text_tokens);
  };

  void Compute(const ortc::Tensor<std::string>& input,
               ortc::Tensor<int64_t>& tokenize_output) const {
    std::vector<std::string> str_input{input.Data()};
    const auto& input_dim = input.Shape();

    size_t max_length = 0;
    std::vector<std::vector<int64_t>> tokenize_results;
    for (auto& str : str_input) {
      auto tokens = tokenizer->encodeBytes(str);
      std::vector<int64_t> tokens_int64(tokens.begin(), tokens.end());
      max_length = std::max(max_length, tokens_int64.size());
      tokenize_results.emplace_back(tokens_int64);
    }

    std::vector<int64_t> output_dim = input_dim;
    output_dim.push_back(max_length);
    auto* token = tokenize_output.Allocate(output_dim);

    int idx = 0;
    for (auto& res : tokenize_results) {
      for (int64_t id : res) {
        token[idx] = id;
        idx++;
      }

      for (size_t i = res.size(); i < max_length; i++) {
        token[idx] = 0;
        idx++;
      }
    }

    for (auto& result : tokenize_results) {
      result.resize(max_length, 0);
    }
  }
};

struct KernelTrieDetokenizer : public BaseKernel {
 private:
  std::shared_ptr<TrieTokenizer> tokenizer;

 public:
  KernelTrieDetokenizer(const OrtApi& api, const OrtKernelInfo& info)
      : BaseKernel(api, info) {
    std::string text_tokens = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab");
    tokenizer = std::make_shared<TrieTokenizer>(text_tokens);
  };

  void Compute(const ortc::Tensor<int64_t>& tokens, ortc::Tensor<std::string>& text) const {
    const int64_t* p_ids = tokens.Data();
    const auto& ids_dim = tokens.Shape();
    std::vector<int64_t> output_dim = {1};
    if (ids_dim.size() > 1) {
      output_dim.resize(ids_dim.size() - 1);
      std::copy(ids_dim.begin(), ids_dim.begin() + ids_dim.size() - 1, output_dim.begin());
    }

    std::vector<std::string> output(output_dim[0]);
    bool failed = false;
    for (auto n = 0; n < output_dim[0]; n++) {
      std::vector<int> ids;
      for (auto i = 0; i < ids_dim[1]; i++) {
        ids.push_back(ort_extensions::narrow<int>(p_ids[n * ids_dim[1] + i]));
      }
      auto raw_string = tokenizer->decodeBytes(ids);
      if (ustring::ValidateUTF8(raw_string)) {
        output[n] = raw_string;
      } else {
        output[n] = "\ufffd";  // bad utf-8 string
        failed = true;
      }
    }

    text.SetStringOutput(output, output_dim);
    if (failed) {
      ORTX_CXX_API_THROW("[KernelTrieDetokenizer] the input ids cannot be parsed as a valid utf-8 string", ORT_RUNTIME_EXCEPTION);
    }
  }
};

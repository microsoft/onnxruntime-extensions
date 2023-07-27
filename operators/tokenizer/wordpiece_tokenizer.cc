// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "wordpiece_tokenizer.hpp"
#include "nlohmann/json.hpp"

KernelWordpieceTokenizer::KernelWordpieceTokenizer(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  // https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/WordpieceTokenizer.md
  // https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/bert_tokenizer.py
  std::string vocab_as_string = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab");
  std::string suffix_indicator = ort_.KernelInfoGetAttribute<std::string>(&info, "suffix_indicator");
  std::string unk = ort_.KernelInfoGetAttribute<std::string>(&info, "unknown_token");
  max_input_chars_per_word_ = TryToGetAttributeWithDefault("max_input_chars_per_word", 200);
  suffix_indicator_ = ustring(suffix_indicator);
  unk_token_ = ustring(unk);

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
  std::unordered_map<std::string, int32_t> vocab_map;
  auto parsed = nlohmann::json::parse(vocab_as_string);
  parsed.get_to(vocab_map);

  for (auto it = vocab_map.begin(); it != vocab_map.end(); ++it) {
    vocab_[ustring(it->first)] = it->second;
  }
}

void KernelWordpieceTokenizer_Split(const std::u32string& /*suffix_indicator*/,
                                    const std::u32string& text,
                                    std::vector<std::u32string>& words) {
  ustring space(" ");
  size_t pos = 0;
  size_t last = 0;
  words.clear();
  for (; pos < text.size(); ++pos) {
    if (text[pos] == space[0]) {
      if (last >= 0 && last < pos) {
        words.push_back(text.substr(last, pos - last));
      }
      last = pos + 1;
    }
  }
  if (last >= 0 && last < text.size()) {
    words.push_back(text.substr(last, pos - last));
  }
}

void KernelWordpieceTokenizer_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
                                        const std::u32string& suffix_indicator,
                                        const ustring& unk_token,
                                        const std::vector<ustring>& texts,
                                        std::vector<ustring>& tokens,
                                        std::vector<int32_t>& indices,
                                        std::vector<int64_t>& rows,
                                        const int64_t* existing_rows,
                                        int64_t n_existing_rows,
                                        int64_t max_input_chars_per_word) {
  std::vector<std::u32string> words;
  bool is_bad;
  bool no_existing_rows = n_existing_rows == 0;
  size_t start = 0, end = 0;
  std::u32string substr;
  int64_t cur_substr;
  tokens.clear();
  indices.clear();
  rows.clear();
  std::u32string token;
  int64_t row_index = 0;
  std::vector<ustring>::const_iterator it;
  int64_t text_index;
  for (it = texts.begin(), text_index = 0; it != texts.end(); ++it, ++text_index) {
    if (no_existing_rows) {
      rows.push_back(indices.size());
    } else if (text_index == existing_rows[row_index]) {
      if (row_index >= n_existing_rows)
        ORTX_CXX_API_THROW(MakeString(
                               "row_index=", row_index, " is out of range=", n_existing_rows, "."),
                           ORT_INVALID_ARGUMENT);
      rows.push_back(indices.size());
      ++row_index;
    }

    KernelWordpieceTokenizer_Split(suffix_indicator, *it, words);

    for (auto itk = words.begin(); itk != words.end(); ++itk) {
      if (static_cast<int64_t>(itk->size()) > max_input_chars_per_word) {
        indices.push_back(-1);
        tokens.push_back(unk_token);
        continue;
      }
      is_bad = false;
      start = 0;
      for (; start < itk->size();) {
        end = itk->size();
        cur_substr = -1;
        for (; start < end;) {
          substr = itk->substr(start, end - start);
          if (start > 0)
            substr = suffix_indicator + substr;
          auto itf = vocab.find(substr);
          if (itf != vocab.end()) {
            token = substr;
            cur_substr = itf->second;
            break;
          }
          end -= 1;
        }
        if (cur_substr == -1) {
          is_bad = true;
          break;
        }
        indices.push_back(static_cast<int32_t>(cur_substr));
        tokens.push_back(ustring(token));
        start = end;
      }
      if (is_bad) {
        indices.push_back(-1);
        tokens.push_back(unk_token);
      }
    }
  }
  rows.push_back(indices.size());
}

void KernelWordpieceTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                       const ortc::Tensor<int64_t>& row_indices,
                                       ortc::Tensor<std::string>& output,
                                       ortc::Tensor<int64_t>& row_lengths,
                                       ortc::Tensor<int64_t>& out_row_begin,
                                       ortc::Tensor<int64_t>& output_limit_values) const {
  // Update with the new API
  // make a copy as we need ustring
  std::vector<ustring> str_input;
  str_input.reserve(input.NumberOfElement());
  for (auto& str : input.Data()) {
    str_input.emplace_back(str);
  }
  const int64_t* p_row_indices = row_indices.Shape().empty() ? nullptr : row_indices.Data();

  std::vector<ustring> tokens;
  std::vector<int32_t> indices;
  std::vector<int64_t> row_begins;

  KernelWordpieceTokenizer_Tokenizer(vocab_, suffix_indicator_, unk_token_, str_input,
                                     tokens, indices, row_begins,
                                     p_row_indices, row_indices.NumberOfElement(),
                                     max_input_chars_per_word_);

  std::vector<int64_t> size_content{(int64_t)indices.size()};
  // TODO: avoid copy
  std::vector<std::string> out_content;
  for (auto& s : tokens)
    out_content.emplace_back(s);
  output.SetStringOutput(out_content, size_content);

  std::vector<int64_t> size_row_lengths{(int64_t)row_begins.size()};
  int64_t* ptr_row_lengths = row_lengths.Allocate(size_row_lengths);
  --size_row_lengths[0];
  int64_t* ptr_row_begins = out_row_begin.Allocate(size_row_lengths);
  int64_t* ptr_limit_values = output_limit_values.Allocate(size_row_lengths);

  int64_t i;
  for (i = 0; i < size_row_lengths[0]; ++i) {
    ptr_row_lengths[i] = row_begins[static_cast<size_t>(i)];
    ptr_row_begins[i] = row_begins[static_cast<size_t>(i)];
    ptr_limit_values[i] = row_begins[static_cast<size_t>(i + 1)];
  }

  i = size_row_lengths[0];
  ptr_row_lengths[i] = row_begins[static_cast<size_t>(i)];
}

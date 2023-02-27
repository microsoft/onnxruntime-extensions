// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Partial code comes from other Microsoft employee.

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <list>
#include <memory>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <functional>
#include <codecvt>
#include <mutex>

#include "nlohmann/json.hpp"
#include "bpetokenizer.hpp"
#include "string_tensor.h"
#include "unicode.h"

// Note: the following logic comes from CPython: unicodetype_db.h (_PyUnicode_IsWhitespace)
bool IsInUnicodeSpace(char32_t ch) {
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

bool IsEmptyUstring(const ustring& str) {
  return std::all_of(str.begin(), str.end(), [](char32_t ch) { return IsInUnicodeSpace(ch); });
}

KernelClipBpeTokenizer::KernelClipBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab");
  if (vocab.empty()) {
    ORTX_CXX_API_THROW("vocabulary shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  std::string merges = ort_.KernelInfoGetAttribute<std::string>(&info, "merges");
  if (merges.empty()) {
    ORTX_CXX_API_THROW("merges shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  if (!TryToGetAttribute<int64_t>("padding_length", padding_length_)) {
    padding_length_ = -1;
  }

  if (padding_length_ != -1 && padding_length_ <= 0) {
    ORTX_CXX_API_THROW("padding_length should be more than 0 or equal -1", ORT_INVALID_ARGUMENT);
  }

  std::stringstream vocabu_stream(vocab);
  std::stringstream merges_stream(merges);
  bbpe_tokenizer_ = std::make_shared<VocabData>();
  bbpe_tokenizer_->Load(vocabu_stream, merges_stream, "<|endoftext|>", "<|endoftext|>");
}

std::vector<int64_t> KernelClipBpeTokenizer::Tokenize(ustring& input, int64_t max_length) {
  std::vector<int64_t> res;

  if (IsEmptyUstring(input)) {
    return res;
  }
  // Add <|startoftext|> token to result
  res.push_back(bbpe_tokenizer_->GetEncoding("<|startoftext|>"));

  // Convert to lowercase
  std::transform(input.begin(), input.end(), input.begin(), [](char32_t c) { return static_cast<char32_t>(ToLower(c)); });

  // Parse input
  auto special_token_split_res = bbpe_tokenizer_->SplitBySpecialTokens(input);
  TokenWithRegularExp regcmp;

  for (auto& seg_id : special_token_split_res) {
    if (static_cast<int64_t>(res.size()) >= max_length) break;

    if (seg_id.second != -1) {
      res.push_back(seg_id.second);
      continue;
    }

    auto cur_input = std::move(seg_id.first);
    // Note: keep ptr to make sure the string_view is valid in the following process
    const char32_t* ptr = cur_input.c_str();
    regcmp.Set(ptr);

    while (static_cast<int64_t>(res.size()) < max_length) {
      auto [b, tok] = regcmp.GetNextToken();
      if (!b) break;

      std::string utf8_token = std::string(ustring(tok));

      // Whitespace clean
      utf8_token.erase(std::remove(utf8_token.begin(), utf8_token.end(), ' '), utf8_token.end());

      // Get byte encodings prior to performing BPE
      byte_list_.clear();
      for (int i = 0; i < utf8_token.length(); i++) {
        if (i == utf8_token.length() - 1) {
          std::string boundary(1, utf8_token[i]);
          byte_list_.push_back(bbpe_tokenizer_->GetEncoding(boundary + "</w>"));
        } else {
          byte_list_.push_back(bbpe_tokenizer_->ByteEncoder()[static_cast<unsigned char>(utf8_token[i])]);
        }
      }

      // Perform BPE
      bbpe_tokenizer_->bpe(byte_list_);

      // Add output to result
      for (auto p : byte_list_) {
        if (static_cast<int64_t>(res.size()) >= max_length) {
          break;
        }

        res.push_back(p);
      }
    }
  }
  // Add <|endoftext|> token to result
  res.push_back(bbpe_tokenizer_->GetEncoding("<|endoftext|>"));
  return res;
}

void KernelClipBpeTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> str_input;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);
  OrtTensorDimensions input_dim(ort_, input);

  std::vector<std::vector<int64_t>> tokenize_results;
  for (auto& str : str_input) {
    ustring ustr = ustring(str);
    tokenize_results.emplace_back(Tokenize(ustr, padding_length_ < 0 ? INT64_MAX : padding_length_));
  }

  size_t max_length = 0;
  if (padding_length_ == -1) {
    for (auto& res : tokenize_results) {
      max_length = std::max(max_length, res.size());
    }
  } else {
    max_length = static_cast<size_t>(padding_length_);
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

    for (size_t i = res.size(); i < max_length; i++) {
      token[idx] = 0;
      mask[idx] = 0;
      idx++;
    }
  }
}

const char* CustomOpClipBpeTokenizer::GetName() const {
  return "CLIPTokenizer";
}

size_t CustomOpClipBpeTokenizer::GetInputTypeCount() const {
  return 1;
}

ONNXTensorElementDataType CustomOpClipBpeTokenizer::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}
size_t CustomOpClipBpeTokenizer::GetOutputTypeCount() const {
  return 2;
}

ONNXTensorElementDataType CustomOpClipBpeTokenizer::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

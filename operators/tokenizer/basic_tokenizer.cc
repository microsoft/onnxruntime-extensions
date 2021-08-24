// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_utils.h"
#include "basic_tokenizer.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

BasicTokenizer::BasicTokenizer(bool do_lower_case, bool tokenize_chinese_chars, bool strip_accents, bool tokenize_punctuation, bool remove_control_chars):
 do_lower_case_(do_lower_case), tokenize_chinese_chars_(tokenize_chinese_chars), strip_accents_(strip_accents), tokenize_punctuation_(tokenize_punctuation),
 remove_control_chars_(remove_control_chars){}

std::vector<ustring> BasicTokenizer::Tokenize(ustring text) {
  std::vector<ustring> result;
  ustring token;
  auto push_current_token_and_clear = [&result, &token]() {
    if (!token.empty()) {
      result.push_back(token);
      token.clear();
    }
  };

  auto push_single_char_and_clear = [&result, &token](char32_t c) {
    token.push_back(c);
    result.push_back(token);
    token.clear();
  };

  // strip accent first
  if (strip_accents_) {
    for (auto& c : text) {
      c = StripAccent(c);
    }
  }

  if (do_lower_case_) {
    for (auto& c : text) {
        c = ::tolower(c);
    }
  }

  for (auto c : text) {
    if (tokenize_chinese_chars_ && IsChineseChar(c)) {
      push_current_token_and_clear();
      push_single_char_and_clear(c);
      continue;
    }

    if (strip_accents_ && IsAccent(c)) {
      continue;
    }

    if (tokenize_punctuation_ && ::ispunct(c)) {
      push_current_token_and_clear();
      push_single_char_and_clear(c);
      continue;
    }

    // split by space
    if (::isspace(c)) {
      push_current_token_and_clear();
      continue;
    }

    // iscntrl will judge \t\f\n\r as control char
    // but it has been filter by isspace(c)
    if (remove_control_chars_ && ::iscntrl(c)) {
      continue;
    }

    token.push_back(c);
  }

  push_current_token_and_clear();
  return result;
}

KernelBasicTokenizer::KernelBasicTokenizer(OrtApi api) : BaseKernel(api) {
}

void KernelBasicTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  auto* output_data = ort_.GetTensorMutableData<int64_t>(output);

  for (int i = 0; i < dimensions.Size(); i++) {
    output_data[i] = ustring(input_data[i]).size();
  }
}

void* CustomOpBasicTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelBasicTokenizer(api);
};

const char* CustomOpBasicTokenizer::GetName() const { return "BasicTokenizer"; };

size_t CustomOpBasicTokenizer::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBasicTokenizer::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpBasicTokenizer::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBasicTokenizer::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

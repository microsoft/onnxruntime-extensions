// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bert_tokenizer.hpp"
#include "kernels/string_common.h"
#include "nlohmann/json.hpp"

KernelBertTokenizer::KernelBertTokenizer(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api) {
  std::string vocab_as_string = ort_.KernelInfoGetAttribute<std::string>(info, "vocab");
  std::string suffix_indicator = ort_.KernelInfoGetAttribute<std::string>(info, "suffix_indicator");
  max_input_chars_per_word_ = HasAttribute("max_input_chars_per_word") ? ort_.KernelInfoGetAttribute<int64_t>(info, "max_input_chars_per_word") : 200;
  suffix_indicator_ = ustring(suffix_indicator);

  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> cvt;
  std::unordered_map<std::string, int32_t> vocab_map;
  auto parsed = nlohmann::json::parse(vocab_as_string);
  parsed.get_to(vocab_map);

  for (auto it = vocab_map.begin(); it != vocab_map.end(); ++it) {
    vocab_[ustring(it->first)] = it->second;
  }
}

void KernelBertTokenizer::Split(const std::u32string& text, std::vector<std::u32string>& words) {
  ustring space(" ");
  int pos = 0;
  int last = -1;
  words.clear();
  for (; pos < text.size(); ++pos) {
    if (text[pos] == space[0]) {
      if (last >= 0 && last < pos) {
        if (words.empty()) {
          words.push_back(text.substr(last, pos - last));
        } else {
          words.push_back(suffix_indicator_ + text.substr(last, pos - last));
        }
      }
      last = pos + 1;
    } else {
      last = pos;
    }
  }
  if (last >= 0 && last < text.size()) {
    words.push_back(ustring(text.substr(last, text.size() - last)));
  }
}

void KernelBertTokenizer::Compute(OrtKernelContext* context) {
  // Update with the new API
  const OrtValue* ort_input = ort_.KernelContext_GetInput(context, 0);
  std::vector<ustring> str_input;
  GetTensorMutableDataString(api_, ort_, context, ort_input, str_input);

  // computation
  // See https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/tokenization.py#L300

  std::vector<std::u32string> words;
  std::vector<int32_t> indices;
  std::vector<int64_t> row_begins;
  row_begins.push_back(0);
  for (auto it = str_input.begin(); it != str_input.end(); ++it) {
    Split(*it, words);
    for (auto itk = words.begin(); itk != words.end(); ++itk) {
      if (itk->size() > max_input_chars_per_word_) {
        indices.push_back(-1);
      } else {
        auto find = vocab_.find(*itk);
        indices.push_back(find == vocab_.end() ? -1 : find->second);
      }
    }
    row_begins.push_back(words.size());
  }

  std::vector<int64_t> size_content(1);
  size_content[0] = indices.size();
  OrtValue* out_content = ort_.KernelContext_GetOutput(context, 0, size_content.data(), size_content.size());

  std::vector<int64_t> size_indices(1);
  size_indices[0] = row_begins.size();
  OrtValue* out_indices = ort_.KernelContext_GetOutput(context, 1, size_indices.data(), size_indices.size());

  int* ptr_content = ort_.GetTensorMutableData<int>(out_content);
  memcpy(ptr_content, indices.data(), indices.size() * sizeof(int));
  int64_t* ptr_indices = ort_.GetTensorMutableData<int64_t>(out_indices);
  memcpy(ptr_indices, row_begins.data(), row_begins.size() * sizeof(int64_t));
}

void* CustomOpBertTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBertTokenizer(api, info);
};

const char* CustomOpBertTokenizer::GetName() const {
  return "BertTokenizer";
};

size_t CustomOpBertTokenizer::GetInputTypeCount() const {
  return 6;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    default:
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpBertTokenizer::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(MakeString("Unexpected output index ", index));
  }
};

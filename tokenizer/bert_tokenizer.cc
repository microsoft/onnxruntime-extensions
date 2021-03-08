// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bert_tokenizer.hpp"
#include "nlohmann/json.hpp"

KernelBertTokenizer::KernelBertTokenizer(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  // https://github.com/tensorflow/text/blob/master/docs/api_docs/python/text/BertTokenizer.md
  // https://github.com/tensorflow/text/blob/master/tensorflow_text/python/ops/bert_tokenizer.py
  std::string vocab_as_string = ort_.KernelInfoGetAttribute<std::string>(info, "vocab");
  std::string suffix_indicator = ort_.KernelInfoGetAttribute<std::string>(info, "suffix_indicator");
  std::string unk = ort_.KernelInfoGetAttribute<std::string>(info, "unk_token");
  max_input_chars_per_word_ = HasAttribute("max_input_chars_per_word") ? ort_.KernelInfoGetAttribute<int64_t>(info, "max_input_chars_per_word") : 200;
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

void KernelBertTokenizer_Split(const std::u32string& suffix_indicator,
                               const std::u32string& text,
                               std::vector<std::u32string>& words) {
  ustring space(" ");
  int pos = 0;
  int last = 0;
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

void KernelBertTokenizer_Tokenizer(const std::unordered_map<std::u32string, int32_t>& vocab,
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
  int start, end;
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
      rows.push_back(indices.size());
      ++row_index;
    }

    KernelBertTokenizer_Split(suffix_indicator, *it, words);

    for (auto itk = words.begin(); itk != words.end(); ++itk) {
      if (itk->size() > max_input_chars_per_word) {
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
        indices.push_back(cur_substr);
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

void KernelBertTokenizer::Compute(OrtKernelContext* context) {
  // Update with the new API
  const OrtValue* ort_input = ort_.KernelContext_GetInput(context, 0);
  std::vector<ustring> str_input;
  GetTensorMutableDataString(api_, ort_, context, ort_input, str_input);
  const OrtValue* ort_row_indices = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions ort_row_indices_dim(ort_, ort_row_indices);
  const int64_t* p_row_indices = ort_row_indices_dim.empty() ? nullptr : ort_.GetTensorData<int64_t>(ort_row_indices);

  std::vector<ustring> tokens;
  std::vector<int32_t> indices;
  std::vector<int64_t> row_begins;
  KernelBertTokenizer_Tokenizer(vocab_, suffix_indicator_, unk_token_, str_input,
                                tokens, indices, row_begins,
                                p_row_indices, ort_row_indices_dim.size(),
                                max_input_chars_per_word_);

  std::vector<int64_t> size_content(1);
  size_content[0] = indices.size();
  OrtValue* out_content = ort_.KernelContext_GetOutput(context, 1, size_content.data(), size_content.size());

  std::vector<int64_t> size_indices(1);
  size_indices[0] = row_begins.size();
  OrtValue* out_indices = ort_.KernelContext_GetOutput(context, 2, size_indices.data(), size_indices.size());

  int* ptr_content = ort_.GetTensorMutableData<int>(out_content);
  memcpy(ptr_content, indices.data(), indices.size() * sizeof(int));
  int64_t* ptr_indices = ort_.GetTensorMutableData<int64_t>(out_indices);
  memcpy(ptr_indices, row_begins.data(), row_begins.size() * sizeof(int64_t));

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, size_content.data(), size_content.size());
  FillTensorDataString(api_, ort_, context, tokens, output);
}

void* CustomOpBertTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBertTokenizer(api, info);
};

const char* CustomOpBertTokenizer::GetName() const {
  return "BertTokenizer";
};

size_t CustomOpBertTokenizer::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpBertTokenizer::GetOutputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(MakeString("Unexpected output index ", index));
  }
};

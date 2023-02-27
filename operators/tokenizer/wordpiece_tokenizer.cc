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
  max_input_chars_per_word_ = HasAttribute("max_input_chars_per_word")
                                  ? ort_.KernelInfoGetAttribute<int64_t>(&info, "max_input_chars_per_word")
                                  : 200;
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

void KernelWordpieceTokenizer::Compute(OrtKernelContext* context) {
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

  KernelWordpieceTokenizer_Tokenizer(vocab_, suffix_indicator_, unk_token_, str_input,
                                     tokens, indices, row_begins,
                                     p_row_indices, ort_row_indices_dim.Size(),
                                     max_input_chars_per_word_);

  std::vector<int64_t> size_content{(int64_t)indices.size()};
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, size_content.data(), size_content.size());
  FillTensorDataString(api_, ort_, context, tokens, output);

  std::vector<int64_t> size_row_lengths{(int64_t)row_begins.size()};
  OrtValue* output_row_lengths = ort_.KernelContext_GetOutput(context, 1, size_row_lengths.data(), size_row_lengths.size());
  --size_row_lengths[0];
  OrtValue* output_row_begins = ort_.KernelContext_GetOutput(context, 2, size_row_lengths.data(), size_row_lengths.size());
  OrtValue* output_limit_values = ort_.KernelContext_GetOutput(context, 3, size_row_lengths.data(), size_row_lengths.size());
  int64_t* ptr_row_lengths = ort_.GetTensorMutableData<int64_t>(output_row_lengths);
  int64_t* ptr_row_begins = ort_.GetTensorMutableData<int64_t>(output_row_begins);
  int64_t* ptr_limit_values = ort_.GetTensorMutableData<int64_t>(output_limit_values);

  int64_t i;
  for (i = 0; i < size_row_lengths[0]; ++i) {
    ptr_row_lengths[i] = row_begins[static_cast<size_t>(i)];
    ptr_row_begins[i] = row_begins[static_cast<size_t>(i)];
    ptr_limit_values[i] = row_begins[static_cast<size_t>(i + 1)];
  }

  i = size_row_lengths[0];
  ptr_row_lengths[i] = row_begins[static_cast<size_t>(i)];
}

const char* CustomOpWordpieceTokenizer::GetName() const {
  return "WordpieceTokenizer";
};

size_t CustomOpWordpieceTokenizer::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpWordpieceTokenizer::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORTX_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
};

size_t CustomOpWordpieceTokenizer::GetOutputTypeCount() const {
  return 4;
};

ONNXTensorElementDataType CustomOpWordpieceTokenizer::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
    case 2:
    case 3:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORTX_CXX_API_THROW(MakeString("[WordpieceTokenizer] Unexpected output index ", index), ORT_INVALID_ARGUMENT);
  }
};

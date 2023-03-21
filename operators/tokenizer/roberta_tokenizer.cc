// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// Partial code comes from other Microsoft employee.

#include "roberta_tokenizer.hpp"
#include "narrow.h"


KernelRobertaBpeTokenizer::KernelRobertaBpeTokenizer(const OrtApi& api, const OrtKernelInfo& info)
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

std::vector<int64_t> KernelRobertaBpeTokenizer::Tokenize(ustring& input, int64_t max_length, std::list<OffsetMappingType>& offset_map) {
  std::vector<int64_t> res;

  if (IsEmptyUString(input)) {
    return res;
  }
  // Add BOS token to result
  res.push_back(bbpe_tokenizer_->GetEncoding("<s>"));

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

    size_t offset = 0;
    OffsetMappingType offset_mapping;

    // Add offset mapping for BOS token
    offset_mapping.push_back(std::make_pair(0, 0));

    while (static_cast<int64_t>(res.size()) < max_length) {
      auto [b, tok] = regcmp.GetNextToken();
      if (!b) break;

      std::string utf8_token = std::string(ustring(tok));

      // Handle special case for offset mapping
      int space_dif = 0;
      if (utf8_token.at(0) == ' ') {
        offset++;
        space_dif = -1; // account for spaces used in offset map algorithm in bpe(byte_list_)
      }

      // Get byte encodings prior to performing BPE
      byte_list_.clear();
      for (char& cp : utf8_token) {
        byte_list_.emplace_back(std::make_pair(bbpe_tokenizer_->ByteEncoder()[static_cast<unsigned char>(cp)], 1));
      }

      // Perform BPE
      bbpe_tokenizer_->bpe(byte_list_);

      // Add output to result
      for (auto p : byte_list_) {
        if (static_cast<int64_t>(res.size()) >= max_length) {
          break;
        }

        res.push_back(p.first);
        offset_mapping.emplace_back(std::make_pair(offset, ort_extensions::narrow<size_t>(offset + p.second + space_dif)));
        offset += (p.second + space_dif);
      }
    }
    // Add offset mapping for EOS token
    offset_mapping.emplace_back(std::make_pair(0, 0));

    // Add offset mappings for input in this instance to list of offset mappings for all inputs
    offset_map.emplace_back(offset_mapping);
  }
  // Add EOS token to result
  res.emplace_back(bbpe_tokenizer_->GetEncoding("</s>"));
  return res;
}

void KernelRobertaBpeTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> str_input;
  std::list<OffsetMappingType> offset_map;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);
  OrtTensorDimensions input_dim(ort_, input);

  std::vector<std::vector<int64_t>> tokenize_results;
  for (auto& str : str_input) {
    ustring ustr = ustring(str);
    tokenize_results.emplace_back(Tokenize(ustr, padding_length_ < 0 ? INT64_MAX : padding_length_, offset_map));
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

  OrtTensorDimensions offset_dim = output_dim;
  offset_dim.push_back(2); // tuple of offsets for each input id

  OrtValue* tokenize_output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
  OrtValue* attention_mask = ort_.KernelContext_GetOutput(context, 1, output_dim.data(), output_dim.size());
  OrtValue* offset_mapping = ort_.KernelContext_GetOutput(context, 2, offset_dim.data(), offset_dim.size());
  auto* token = ort_.GetTensorMutableData<int64_t>(tokenize_output);
  auto* mask = ort_.GetTensorMutableData<int64_t>(attention_mask);
  auto* offset = ort_.GetTensorMutableData<int64_t>(offset_mapping);

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

const char* CustomOpRobertaBpeTokenizer::GetName() const {
  return "RobertaTokenizer";
}

size_t CustomOpRobertaBpeTokenizer::GetInputTypeCount() const {
  return 1;
}

ONNXTensorElementDataType CustomOpRobertaBpeTokenizer::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
}
size_t CustomOpRobertaBpeTokenizer::GetOutputTypeCount() const {
  return 3;
}

ONNXTensorElementDataType CustomOpRobertaBpeTokenizer::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
}

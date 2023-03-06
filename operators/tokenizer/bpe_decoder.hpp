// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "ustring.h"
#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <set>
#include <map>


struct KernelBpeDecoder : public BaseKernel {
 public:
  KernelBpeDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "id_vocab");
    if (vocab.empty()) {
      ORTX_CXX_API_THROW("id vocab text cannot be empty.", ORT_INVALID_ARGUMENT);
    }

    std::string byte_decoder = ort_.KernelInfoGetAttribute<std::string>(&info, "byte_decoder");
    if (byte_decoder.empty()) {
      ORTX_CXX_API_THROW("vocabulary shouldn't be empty.", ORT_INVALID_ARGUMENT);
    }

    std::string added_tokens = ort_.KernelInfoGetAttribute<std::string>(&info, "added_tokens");
    if (added_tokens.empty()) {
      ORTX_CXX_API_THROW("merges shouldn't be empty.", ORT_INVALID_ARGUMENT);
    }

    en_normalization_ = TryToGetAttributeWithDefault<int64_t>("en_normalization", 0);
    BuildIdVocab(vocab);
  }

  void BuildIdVocab(const std::string& vocab) {
    arr_vocab_.reserve(vocab.size() / 2 ); // give a rough estimation.

    ustring u_vocab(vocab);
    std::u32string_view uv_vocab;
    size_t last_pos = 0;

    arr_vocab_.emplace_back(ustring("")); // Id starts from 1
    for (size_t n = 0; n < uv_vocab.size(); ++n) {
      if (uv_vocab[n] == char32_t('\n')) {
        std::u32string_view s_tok = uv_vocab.substr(last_pos, n - last_pos);
        arr_vocab_.emplace_back(ustring(s_tok));
        last_pos = n + 1;
      }
    }

    arr_vocab_.shrink_to_fit();
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* ids = ort_.KernelContext_GetInput(context, 0);

    const OrtValue* byte_decoder = ort_.KernelContext_GetInput(context, 1);
    const OrtValue* added_tokens = ort_.KernelContext_GetInput(context, 2);
    const int64_t* p_ids = ort_.GetTensorData<int64_t>(ids);
    OrtTensorDimensions ids_dim(ort_, ids);

    if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
      ORTX_CXX_API_THROW("[BpeDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
    }

    std::string text;
    bool f_special_last = false;
    bool f_special = false;
    auto count = static_cast<size_t>(ids_dim.Size());

    for (size_t tok_idx = 0; tok_idx < count; ++tok_idx) {
      const auto token = *(p_ids + tok_idx);
      std::string decoded_token;
      f_special = all_special_ids_.count(token)? true : false;
      if (skip_special_tokens_){
        continue;
      }

      if (added_tokens_.count(token)) {
        const std::string ws = added_tokens_.at(token);
        decoded_token = (std::string)ws;
      }
      else {
        const auto str = arr_vocab_[token];
        // string is a regular token from known vocab
        for (auto wchr : str) {
          // get output character from byte decoder for each wide character
          unsigned char uchr = byte_decoder_.at(wchr);
          decoded_token.push_back(uchr);
        }
      }

      if (f_special && (tok_idx > 0 && !f_special_last)) {
        text.push_back(' ');
      }

      text.append(decoded_token);
      if (f_special && tok_idx + 1 != count) {
        text.push_back(' ');
      }

      f_special_last = f_special;
    }
  }

 private:
  // Since ORT API doesn't support boolean type in ONNX node attribute,
  // all flag attributes here are defined as int64 type to be more explicit.
  int64_t en_normalization_ = 0;
  int64_t skip_special_tokens_ = 0;
  std::vector<ustring> arr_vocab_;
  std::map<char32_t, unsigned char> byte_decoder_;
  std::map<int64_t, std::string> added_tokens_;
  std::set<int64_t> all_special_ids_;
};


struct CustomOpBpeDecoder : OrtW::CustomOpBase<CustomOpBpeDecoder, KernelBpeDecoder> {
  const char* GetName() const {
    return "BpeDecoder";
  }

  size_t GetInputTypeCount() const {
    return 3;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    ONNXTensorElementDataType input_types[] = {
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 };

    if (index >= sizeof(input_types) / sizeof(ONNXTensorElementDataType)) {
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    return input_types[index];
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
};

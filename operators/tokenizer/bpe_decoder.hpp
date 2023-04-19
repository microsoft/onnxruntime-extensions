// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "ustring.h"
#include "narrow.h"
#include <string>
#include <vector>
#include <locale>
#include <codecvt>
#include <set>
#include <map>
#include <unordered_map>

struct KernelBpeDecoder : public BaseKernel {
 public:
  KernelBpeDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "id_vocab");
    if (vocab.empty()) {
      ORTX_CXX_API_THROW("[BPEDecoder]id vocab text cannot be empty.", ORT_INVALID_ARGUMENT);
    }
    BuildIdVocab(vocab);

    std::string byte_decoder = ort_.KernelInfoGetAttribute<std::string>(&info, "byte_decoder");
    if (byte_decoder.empty()) {
      ORTX_CXX_API_THROW("[BPEDecoder]byte_decoder cannot be empty.", ORT_INVALID_ARGUMENT);
    } else {
      auto um = ParseId2String(byte_decoder);
      std::transform(um.begin(), um.end(),
                     std::inserter(byte_decoder_, byte_decoder_.end()),
                     [](const auto& p) { return std::make_pair(static_cast<char32_t>(p.first),
                                                               ort_extensions::narrow<unsigned char>(std::stoul(p.second))); });
    }

    std::string added_tokens = TryToGetAttributeWithDefault<std::string>("added_tokens", "");
    if (!added_tokens.empty()) {
      auto um = ParseId2String(added_tokens);
      added_tokens_ = std::map<int64_t, std::string>(um.begin(), um.end());
    }

    std::string all_special_ids = TryToGetAttributeWithDefault<std::string>("all_special_ids", "");
    if (!all_special_ids.empty()) {
      auto um = ParseId2String(all_special_ids);
      std::transform(um.begin(), um.end(),
                     std::inserter(all_special_ids_, all_special_ids_.end()),
                     [](const auto& p) { return p.first; });
    }

    en_normalization_ = TryToGetAttributeWithDefault<int64_t>("en_normalization", 0);
    skip_special_tokens_ = TryToGetAttributeWithDefault<int64_t>("skip_special_tokens", 0);
    whitespace_token_ = TryToGetAttributeWithDefault<int64_t>("whitespace_token", 0);
    bos_token_ = TryToGetAttributeWithDefault("bos_token", std::string("<|endoftext|>"));
    eos_token_ = TryToGetAttributeWithDefault("eos_token", std::string("<|endoftext|>"));
    unk_token_ = TryToGetAttributeWithDefault("unk_token", std::string("<|endoftext|>"));
  }

  std::unordered_map<int64_t, std::string> ParseId2String(const std::string& s_attr) {
    std::unordered_map<int64_t, std::string> result;
    result.reserve(s_attr.size() / 4);
    std::stringstream ss(s_attr);

    std::string line;
    std::string token;
    while (std::getline(ss, line, '\n')) {
      size_t pos_end = 0;
      int64_t v = std::stoll(line, &pos_end);
      if (pos_end >= line.size() || line[pos_end] != '\t') {
        token.clear();
      } else {
        token = line.substr(pos_end + 1);
      }
      result.emplace(v, token);
    }

    return result;
  }

  void BuildIdVocab(const std::string& vocab) {
    arr_vocab_.reserve(vocab.size() / 2);  // give a rough estimation.

    std::u32string u_vocab = ustring(vocab);
    std::u32string_view uv_vocab(u_vocab);
    size_t last_pos = 0;

    auto ccount = uv_vocab.size();
    for (size_t n = 0; n < ccount; ++n) {
      if (uv_vocab[n] == char32_t('\n')) {
        std::u32string_view s_tok = uv_vocab.substr(last_pos, n - last_pos);
        arr_vocab_.emplace_back(ustring(s_tok));
        last_pos = n + 1;
      } else if (n == ccount - 1) {
        std::u32string_view s_tok = uv_vocab.substr(last_pos, n - last_pos + 1);
        arr_vocab_.emplace_back(ustring(s_tok));
      }
    }

    arr_vocab_.shrink_to_fit();
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* ids = ort_.KernelContext_GetInput(context, 0);
    const int64_t* p_ids = ort_.GetTensorData<int64_t>(ids);
    OrtTensorDimensions ids_dim(ort_, ids);
    std::vector<int64_t> output_dim = {1};
    if (ids_dim.size() > 1) {
      output_dim.resize(ids_dim.size() - 1);
      std::copy(ids_dim.begin(), ids_dim.begin() + ids_dim.size() - 1, output_dim.begin());
    }

    size_t seq_len = ids_dim.back();
    size_t string_batch = ids_dim.Size() / seq_len;
    std::vector<std::string> decoded_strings;
    decoded_strings.reserve(string_batch);
    for (auto n = string_batch; n > 0; n--) {
      std::string text;
      bool f_special_last = false;
      bool f_special = false;
      auto count = static_cast<size_t>(ids_dim.Size());

      for (size_t tok_idx = 0; tok_idx < count; ++tok_idx) {
        const auto token = *(p_ids + tok_idx);
        std::string decoded_token;
        f_special = all_special_ids_.count(token) ? true : false;
        if (skip_special_tokens_ && f_special) {
          f_special_last = f_special;
          continue;
        }

        if (added_tokens_.count(token)) {
          const std::string ws = added_tokens_.at(token);
          decoded_token = (std::string)ws;
        } else if (static_cast<size_t>(token) < arr_vocab_.size()) {
          const auto str = arr_vocab_[token];
          for (auto wchr : str) {
            unsigned char uchr = byte_decoder_.at(wchr);
            decoded_token.push_back(uchr);
          }
        } else {
          if (skip_special_tokens_) {
            continue;
          } else {
            decoded_token = unk_token_;
          }
        }

        if (whitespace_token_ &&
            f_special && (tok_idx > 0 && !f_special_last)) {
          text.push_back(' ');
        }

        text.append(decoded_token);

        if (whitespace_token_ &&
            f_special && tok_idx != count - 1) {
          text.push_back(' ');
        }

        f_special_last = f_special;
      }

      decoded_strings.emplace_back(std::move(text));
      p_ids += seq_len;
    }

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
    FillTensorDataString(api_, ort_, context, decoded_strings, output);
  }

 private:
  std::string bos_token_;
  std::string eos_token_;
  std::string unk_token_;

  // Since ORT API doesn't support boolean type in ONNX node attribute,
  // all flag attributes here are defined as int64 type to be more explicit.
  int64_t en_normalization_ = 0;
  int64_t skip_special_tokens_ = 0;
  int64_t whitespace_token_ = 0;
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
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
};

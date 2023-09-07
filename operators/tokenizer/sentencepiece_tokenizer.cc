// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_tokenizer.hpp"
#include "string_tensor.h"
#include "base64.h"

KernelSentencepieceTokenizer::KernelSentencepieceTokenizer(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info) {
  std::string model_as_string = ort_.KernelInfoGetAttribute<std::string>(&info, "model");
  sentencepiece::ModelProto model_proto;
  std::vector<uint8_t> model_as_bytes;
  if (base64_decode(model_as_string, model_as_bytes)) {
    model_proto.ParseFromArray(model_as_bytes.data(), static_cast<int>(model_as_bytes.size()));
  } else {
    model_proto.ParseFromArray(model_as_string.c_str(), static_cast<int>(model_as_string.size()));
  }
  sentencepiece::util::Status status = tokenizer_.Load(model_proto);
  if (!status.ok())
    ORTX_CXX_API_THROW(MakeString("Failed to create SentencePieceProcessor instance. Error code is ",
                                  (int)status.code(), ". Message is '", status.error_message(), "'."),
                       ORT_FAIL);
}

void KernelSentencepieceTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                           int64_t /*nbest_size*/,
                                           float /*alpha*/,
                                           bool add_bos,
                                           bool add_eos,
                                           bool add_rev,
                                           ortc::Tensor<int32_t>& output,
                                           ortc::Tensor<int64_t>& output1,
                                           std::optional<bool> xlm_roberta) const {
  // Update with the new API
  auto& str_input = input.Data();
  // computation

  std::vector<int64_t> indices;
  std::vector<int> content;
  indices.reserve(str_input.size() + 1);
  for (size_t i = 0; i < str_input.size(); ++i) {
    std::vector<int> inloop;
    if (!tokenizer_.Encode(str_input[i].c_str(), &inloop).ok())
      ORTX_CXX_API_THROW(MakeString("Unable to encode string '", str_input[i], "'."), ORT_INVALID_ARGUMENT);
    indices.push_back(content.size());

    if (add_rev) {
      if (add_eos) {
        content.push_back(tokenizer_.eos_id());
      }
      content.insert(content.end(), inloop.rbegin(), inloop.rend());
      if (add_bos) {
        content.push_back(tokenizer_.bos_id());
      }
    } else if (xlm_roberta.has_value() && xlm_roberta) {
      // As per HF, the first "real" token "," has position 4 in the XLMRobertaTokenizer vocab
      // and position 3 in the SPM vocab, so we add a padding value of 1 to all IDs.
      //std::transform(content.begin(), content.end(), content.begin(), [](int& i) { return i + 1; });
      std::for_each(inloop.begin(), inloop.end(), [](int& n) { n++; });

      // Add HF BOS (0) and EOS (2) tokens for the XLMRobertaTokenizer
      content.push_back(xlm_bos);
      content.insert(content.end(), inloop.begin(), inloop.end());
      content.push_back(xlm_eos);
    } else {
      if (add_bos) {
        content.push_back(tokenizer_.bos_id());
      }
      content.insert(content.end(), inloop.begin(), inloop.end());
      if (add_eos) {
        content.push_back(tokenizer_.eos_id());
      }
    }
  }
  indices.push_back(content.size());

  // Setup output
  std::vector<int64_t> size_content(1);
  size_content[0] = content.size();

  std::vector<int64_t> size_indices(1);
  size_indices[0] = indices.size();

  int* ptr_content = output.Allocate(size_content);
  memcpy(ptr_content, content.data(), content.size() * sizeof(int));
  int64_t* ptr_indices = output1.Allocate(size_indices);
  memcpy(ptr_indices, indices.data(), indices.size() * sizeof(int64_t));
}

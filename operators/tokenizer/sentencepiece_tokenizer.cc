// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece.pb.h"
#include "sentencepiece_tokenizer.hpp"
#include "string_tensor.h"
#include "base64.h"
#include "narrow.h"

OrtStatusPtr KernelSentencepieceTokenizer::OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
  std::string model_as_string;
  ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "model", model_as_string));

  sentencepiece::ModelProto model_proto;
  std::vector<uint8_t> model_as_bytes;
  if (base64_decode(model_as_string, model_as_bytes)) {
    model_proto.ParseFromArray(model_as_bytes.data(), static_cast<int>(model_as_bytes.size()));
  } else {
    model_proto.ParseFromArray(model_as_string.c_str(), static_cast<int>(model_as_string.size()));
  }
  sentencepiece::util::Status spm_status = tokenizer_.Load(model_proto);
  if (!spm_status.ok()) {
    return OrtW::CreateStatus(MakeString("Failed to create SentencePieceProcessor instance. Error code is ",
                                         (int)spm_status.code(), ". Message is '", spm_status.error_message(), "'."),
                              ORT_FAIL);
  }

  return nullptr;
}

OrtStatusPtr KernelSentencepieceTokenizer::Compute(const ortc::Tensor<std::string>& input,
                                                   int64_t /*nbest_size*/,
                                                   float /*alpha*/,
                                                   bool add_bos,
                                                   bool add_eos,
                                                   bool add_rev,
                                                   ortc::Tensor<int32_t>& output,
                                                   ortc::Tensor<int64_t>& output1,
                                                   std::optional<bool> fairseq,
                                                   std::optional<ortc::Tensor<int32_t>*> output2) const {
  // Update with the new API
  auto& str_input = input.Data();
  // computation

  std::vector<int64_t> instance_indices;
  std::vector<int32_t> token_indices;
  std::vector<int> content;
  sentencepiece::SentencePieceText spt;
  instance_indices.reserve(str_input.size() + 1);
  for (size_t i = 0; i < str_input.size(); ++i) {
    if (!tokenizer_.Encode(str_input[i].c_str(), &spt).ok()) {
      return OrtW::CreateStatus(MakeString("Unable to encode string '", str_input[i], "'."), ORT_INVALID_ARGUMENT);
    }

    instance_indices.push_back(content.size());

    if (add_rev) {
      if (add_eos) {
        content.push_back(tokenizer_.eos_id());
        token_indices.push_back(ort_extensions::narrow<int32_t>(str_input[i].length()));
      }
      const auto& pieces = spt.pieces();
      for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) {
        content.push_back((*it).id());
        token_indices.push_back((*it).begin());
      }
      if (add_bos) {
        content.push_back(tokenizer_.bos_id());
        token_indices.push_back(0);
      }
    } else {
      if (add_bos) {
        content.push_back(tokenizer_.bos_id());
        token_indices.push_back(0);
      }
      for (const auto& sp : spt.pieces()) {
        content.push_back(sp.id());
        token_indices.push_back(sp.begin());
      }
      if (add_eos) {
        content.push_back(tokenizer_.eos_id());
        token_indices.push_back(ort_extensions::narrow<int32_t>(str_input[i].length()));
      }
    }
  }
  instance_indices.push_back(content.size());

  // Patch fairseq indices
  if (fairseq.has_value() && (*fairseq) && !add_rev) {
    // HF Fairseq Example (XLMRobertaTokenizer) : https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/xlm_roberta/tokenization_xlm_roberta.html#XLMRobertaTokenizer
    //
    // Original fairseq vocab and spm vocab must be "aligned":
    // Vocab    |    0    |    1    |    2    |    3    |  4  |  5  |  6  |   7   |   8   | 9
    // -------- | ------- | ------- | ------  | ------- | --- | --- | --- | ----- | ----- | ----
    // fairseq  | '<s>'   | '<pad>' | '</s>'  | '<unk>' | ',' | '.' | '▁' | 's'   | '▁de' | '-'
    // spm      | '<unk>' | '<s>'   | '</s>'  | ','     | '.' | '▁' | 's' | '▁de' | '-'   | '▁a'
    //
    // As per HF, the first "real" token "," has position 4 in the XLMRobertaTokenizer vocab and position
    // 3 in the SPM vocab, so we add a padding value of 1 to IDs, and fix exceptions for '<unk>' and '<s>'.
    std::for_each(content.begin(), content.end(), [](int& n) {
      if (n == 0) {  // '<unk>': 0 -> 3
        n = 3;
      } else if (n == 1) {  // '<s>': 1 -> 0
        n = 0;
      } else if (n != 2) {  // '</s>': 2 -> 2, '<*>': x -> x + 1
        n++;
      }
    });
  }

  // Setup output
  std::vector<int64_t> size_content(1);
  size_content[0] = content.size();

  std::vector<int64_t> size_instance_indices(1);
  size_instance_indices[0] = instance_indices.size();

  int* ptr_content = output.Allocate(size_content);
  memcpy(ptr_content, content.data(), content.size() * sizeof(int));
  int64_t* ptr_instance_indices = output1.Allocate(size_instance_indices);
  memcpy(ptr_instance_indices, instance_indices.data(), instance_indices.size() * sizeof(int64_t));
  if (output2.has_value()) {
    int32_t* ptr_token_indices = (*output2)->Allocate(size_content);
    memcpy(ptr_token_indices, token_indices.data(), token_indices.size() * sizeof(int32_t));
  }

  return nullptr;
}

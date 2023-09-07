// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "sentencepiece_processor.h"

struct KernelSentencepieceTokenizer : BaseKernel {
  KernelSentencepieceTokenizer(const OrtApi& api, const OrtKernelInfo& info);
  void Compute(const ortc::Tensor<std::string>& input,
               int64_t /*nbest_size*/,
               float /*alpha*/,
               bool add_bos,
               bool add_eos,
               bool add_rev,
               ortc::Tensor<int32_t>& output,
               ortc::Tensor<int64_t>& output1,
               std::optional<bool> xlm_roberta) const;

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
  // current HF ids for BOS and EOS tokens for XLMRobertaTokenizer
  int xlm_bos = 0;
  int xlm_eos = 2;
};

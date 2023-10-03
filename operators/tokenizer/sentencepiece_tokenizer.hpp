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
               std::optional<bool> fairseq,
               std::optional<ortc::Tensor<int32_t>*> output2) const;

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "sentencepiece_processor.h"


struct KernelSentencepieceTokenizer : BaseKernel {
  KernelSentencepieceTokenizer(const OrtApi& api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

struct CustomOpSentencepieceTokenizer : Ort::CustomOpBase<CustomOpSentencepieceTokenizer, KernelSentencepieceTokenizer> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "kernels/kernels.h"
#include "utils/string_utils.h"
#include "sentencepiece_processor.h"

struct KernelSentencepieceTokenizer : BaseKernel {
  KernelSentencepieceTokenizer(OrtApi api, const OrtKernelInfo* info);
  void Compute(OrtKernelContext* context);

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

struct CustomOpSentencepieceTokenizer : Ort::CustomOpBase<CustomOpSentencepieceTokenizer, KernelSentencepieceTokenizer> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* info) const;
  const char* GetName() const;
  size_t GetInputTypeCount() const;
  ONNXTensorElementDataType GetInputType(size_t index) const;
  size_t GetOutputTypeCount() const;
  ONNXTensorElementDataType GetOutputType(size_t index) const;
};

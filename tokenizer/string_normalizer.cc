// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalizer.hpp"
#include "kernels/string_common.h"
#include "utils/string_utils.h"
#include "sentencepiece_trainer.h"
#include "normalizer.h"
#include <vector>
#include <cmath>
#include <algorithm>

KernelStringNormalizer::KernelStringNormalizer(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string form;
  if (HasAttribute("form")) {
    form = ort_.KernelInfoGetAttribute<std::string>(info_, "form");
  }
  sentencepiece::NormalizerSpec spec;
  if (form == "NFKC")
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("nfkc");
  else if (form == "NFKC_CF")
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("nfkc_cf");
  else if (form == "NMT_NFKC")
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc");
  else if (form == "NMT_NFKC_CF")
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc_cf");
  else if (form.empty() || (form == "ID"))
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("identity");
  else
    throw std::runtime_error(MakeString("Unexpected value for form '", form, "'."));
  normalizer_ = (void*)new sentencepiece::normalizer::Normalizer(spec);
}

KernelStringNormalizer::~KernelStringNormalizer() {
  delete (sentencepiece::normalizer::Normalizer*)normalizer_;
}

void KernelStringNormalizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> X;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  sentencepiece::normalizer::Normalizer* type_normalizer = ((sentencepiece::normalizer::Normalizer*)normalizer_);
  // Do computation
  for (int64_t i = 0; i < (int64_t)X.size(); ++i) {
    X[i] = type_normalizer->Normalize(X[i]);
  }

  // Fills the output
  FillTensorDataString(api_, ort_, context, X, output);
}

void* CustomOpStringNormalizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelStringNormalizer(api, info);
};

const char* CustomOpStringNormalizer::GetName() const { return "StringNormalizer"; };

size_t CustomOpStringNormalizer::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringNormalizer::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringNormalizer::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringNormalizer::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

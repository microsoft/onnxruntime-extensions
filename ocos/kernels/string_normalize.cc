// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_normalize.hpp"
#include "string_common.h"
#include "sentencepiece_trainer.h"
#include <vector>
#include <cmath>
#include <algorithm>

KernelStringNormalize::KernelStringNormalize(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string form;
  if (HasAttribute("form")) {
    form = ort_.KernelInfoGetAttribute<std::string>(info, "form");
  }
  sentencepiece::NormalizerSpec spec;
  if (form.empty() || (form == "NFKC"))
    spec = sentencepiece::SentencePieceTrainer::GetNormalizerSpec("nmt_nfkc");
  else
    throw std::runtime_error(MakeString("Unexpected value for form '", form, "'."));
  normalizer_ = new sentencepiece::normalizer::Normalizer(spec);
}

KernelStringNormalize::~KernelStringNormalize() {
  delete normalizer_;
}

void KernelStringNormalize::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> X;
  GetTensorMutableDataString(api_, ort_, context, input_X, X);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input_X);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());

  // Do computation
  for (int64_t i = 0; i < (int64_t)X.size(); ++i) {
    X[i] = normalizer_->Normalize(X[i]);
  }

  // Fills the output
  FillTensorDataString(api_, ort_, context, X, output);
}

void* CustomOpStringNormalize::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringNormalize(api);
};

const char* CustomOpStringNormalize::GetName() const { return "StringNormalize"; };

size_t CustomOpStringNormalize::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringNormalize::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpStringNormalize::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringNormalize::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

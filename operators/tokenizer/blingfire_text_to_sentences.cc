// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "blingfire_text_to_sentences.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>


KernelTextToSentences::KernelTextToSentences(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string model_data = ort_.KernelInfoGetAttribute<std::string>(info, "model");
  if (model_data.empty()) {
    throw std::runtime_error("vocabulary shouldn't be empty.");
  }

  void* model_ptr = SetModel(reinterpret_cast<unsigned char*>(model_data.data()), model_data.size());

  if (model_ptr == nullptr) {
    throw std::runtime_error("Invalid model");
  }

  model_ = std::shared_ptr<void>(model_ptr, FreeModel);
}

void KernelTextToSentences::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  OrtTensorDimensions dimensions(ort_, input);

  if (dimensions.Size() != 1) {
    throw std::runtime_error("We only support one text.");
  }

  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);


  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  auto* output_data = ort_.GetTensorMutableData<int64_t>(output);

}

void* CustomOpTextToSentences::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelTextToSentences(api, info);
};

const char* CustomOpTextToSentences::GetName() const { return "TextToSentence"; };

size_t CustomOpTextToSentences::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTextToSentences::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpTextToSentences::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpTextToSentences::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

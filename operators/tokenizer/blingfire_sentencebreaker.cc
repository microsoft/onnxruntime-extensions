// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "blingfire_sentencebreaker.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>

KernelBlingFireSentenceBreaker::KernelBlingFireSentenceBreaker(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info), max_sentence(-1) {
  model_data_ = ort_.KernelInfoGetAttribute<std::string>(info, "model");
  if (model_data_.empty()) {
    throw std::runtime_error("vocabulary shouldn't be empty.");
  }

  void* model_ptr = SetModel(reinterpret_cast<unsigned char*>(model_data_.data()), model_data_.size());

  if (model_ptr == nullptr) {
    throw std::runtime_error("Invalid model");
  }

  model_ = std::shared_ptr<void>(model_ptr, FreeModel);

  if (HasAttribute("max_sentence")) {
    max_sentence = ort_.KernelInfoGetAttribute<int64_t>(info, "max_sentence");
  }
}

void KernelBlingFireSentenceBreaker::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  OrtTensorDimensions dimensions(ort_, input);

  if (dimensions.Size() != 1 && dimensions[0] != 1) {
    throw std::runtime_error("We only support string scalar.");
  }

  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  std::string& input_string = input_data[0];
  int max_length = 2 * input_string.size() + 1;
  std::string output_str;
  output_str.reserve(max_length);

  int output_length = TextToSentencesWithOffsetsWithModel(input_string.data(), input_string.size(), output_str.data(), nullptr, nullptr, max_length, model_.get());
  if (output_length < 0) {
    throw std::runtime_error(MakeString("splitting input:\"", input_string, "\"  failed"));
  }

  // inline split output_str by newline '\n'
  std::vector<char*> output_sentences;
  bool head_flag = true;
  for (int i = 0; i < output_length; i++) {
    if (head_flag) {
      output_sentences.push_back(&output_str[i]);
      head_flag = false;
    }

    if (output_str[i] == '\n') {
      head_flag = true;
      output_str[i] = '\0';
    }
  }

  std::vector<int64_t> output_dimensions(1);
  output_dimensions[0] = output_sentences.size();

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dimensions.data(), output_dimensions.size());
  Ort::ThrowOnError(api_, api_.FillStringTensor(output, output_sentences.data(), output_sentences.size()));
}

void* CustomOpBlingFireSentenceBreaker::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBlingFireSentenceBreaker(api, info);
};

const char* CustomOpBlingFireSentenceBreaker::GetName() const { return "BlingFireSentenceBreaker"; };

size_t CustomOpBlingFireSentenceBreaker::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBlingFireSentenceBreaker::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpBlingFireSentenceBreaker::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBlingFireSentenceBreaker::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

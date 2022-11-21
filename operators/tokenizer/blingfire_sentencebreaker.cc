// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "blingfire_sentencebreaker.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <memory>

KernelBlingFireSentenceBreaker::KernelBlingFireSentenceBreaker(const OrtApi& api, const OrtKernelInfo* info) : BaseKernel(api, info), max_sentence(-1) {
  model_data_ = ort_.KernelInfoGetAttribute<std::string>(info, "model");
  if (model_data_.empty()) {
    ORT_CXX_API_THROW("vocabulary shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  void* model_ptr = SetModel(reinterpret_cast<const unsigned char*>(model_data_.data()), static_cast<int>(model_data_.size()));

  if (model_ptr == nullptr) {
    ORT_CXX_API_THROW("Invalid model", ORT_INVALID_ARGUMENT);
  }

  model_ = std::shared_ptr<void>(model_ptr, FreeModel);

  if (HasAttribute("max_sentence")) {
    max_sentence = static_cast<int>(ort_.KernelInfoGetAttribute<int64_t>(info, "max_sentence"));
  }
}

void KernelBlingFireSentenceBreaker::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  OrtTensorDimensions dimensions(ort_, input);

  // TODO: fix this scalar check.
  if (dimensions.Size() != 1 && dimensions[0] != 1) {
    ORT_CXX_API_THROW("We only support string scalar.", ORT_INVALID_ARGUMENT);
  }

  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  std::string& input_string = input_data[0];
  int max_length = static_cast<int>(2 * input_string.size() + 1);
  std::unique_ptr<char[]> output_str = std::make_unique<char[]>(max_length);

  int output_length = TextToSentencesWithOffsetsWithModel(input_string.data(), static_cast<int>(input_string.size()), output_str.get(), nullptr, nullptr, max_length, model_.get());
  if (output_length < 0) {
    ORT_CXX_API_THROW(MakeString("splitting input:\"", input_string, "\"  failed"), ORT_INVALID_ARGUMENT);
  }

  // inline split output_str by newline '\n'
  std::vector<const char*> output_sentences;

  if (output_length == 0) {
    // put one empty string if output_length is 0
    output_sentences.push_back("");
  } else {
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
  }

  std::vector<int64_t> output_dimensions(1);
  output_dimensions[0] = output_sentences.size();

  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dimensions.data(), output_dimensions.size());
  Ort::ThrowOnError(api_, api_.FillStringTensor(output, output_sentences.data(), output_sentences.size()));
}

void* CustomOpBlingFireSentenceBreaker::CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
  return CreateKernelImpl(api, info);
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

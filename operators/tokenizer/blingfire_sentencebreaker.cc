// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "blingfire_sentencebreaker.hpp"
#include "string_tensor.h"
#include <vector>
#include <locale>
#include <codecvt>
#include <algorithm>
#include <memory>

KernelBlingFireSentenceBreaker::KernelBlingFireSentenceBreaker(const OrtApi& api, const OrtKernelInfo& info)
    : BaseKernel(api, info), max_sentence(-1) {
  model_data_ = ort_.KernelInfoGetAttribute<std::string>(&info, "model");
  if (model_data_.empty()) {
    ORTX_CXX_API_THROW("vocabulary shouldn't be empty.", ORT_INVALID_ARGUMENT);
  }

  void* model_ptr = SetModel(reinterpret_cast<const unsigned char*>(model_data_.data()), static_cast<int>(model_data_.size()));

  if (model_ptr == nullptr) {
    ORTX_CXX_API_THROW("Invalid model", ORT_INVALID_ARGUMENT);
  }

  model_ = std::shared_ptr<void>(model_ptr, FreeModel);

  max_sentence = TryToGetAttributeWithDefault("max_sentence", -1);
}

void KernelBlingFireSentenceBreaker::Compute(const std::string& input,
                                             ortc::Tensor<std::string>& output) {
  int max_length = static_cast<int>(2 * input.size() + 1);
  std::unique_ptr<char[]> output_str = std::make_unique<char[]>(max_length);

  int output_length = TextToSentencesWithOffsetsWithModel(input.data(), static_cast<int>(input.size()), output_str.get(), nullptr, nullptr, max_length, model_.get());
  if (output_length < 0) {
    ORTX_CXX_API_THROW(MakeString("splitting input:\"", input, "\"  failed"), ORT_INVALID_ARGUMENT);
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
  output.SetStringOutput(output_sentences, output_dimensions);
}

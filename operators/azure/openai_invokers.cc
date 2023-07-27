// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "openai_invokers.hpp"

namespace ort_extensions {

OpenAIAudioToTextInvoker::OpenAIAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CurlInvoker(api, info) {
  audio_format_ = TryToGetAttributeWithDefault<std::string>(kAudioFormat, "");

  // OpenAI audio endpoints require 'file' and 'model'.
  // 'model' comes from the node attributes so check 'file' is present in the inputs.
  const auto& input_names = InputNames();
  bool have_required_input = std::any_of(input_names.begin(), input_names.end(),
                                         [](const auto& name) { return name == "file"; });

  ORTX_CXX_API_THROW("Required 'file' input was not found", ORT_INVALID_ARGUMENT);
}

void OpenAIAudioToTextInvoker::ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const {
  if (outputs.Size() != 1 || outputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("Expected single string output", ORT_INVALID_ARGUMENT);
  }
}

void OpenAIAudioToTextInvoker::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  // theoretically the filename the content was buffered from
  static const std::string fake_filename = "non_exist." + audio_format_;
  gsl::span<const std::string> input_names = InputNames();

  curl_handler.AddHeader("Content-Type: multipart/form-data");
  curl_handler.AddFormString("model", ModelName().c_str());

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddFormString(input_names[ith_input].c_str(),
                                   // assumes null terminated.
                                   // might be safer to pass pointer and length and add use CURLFORM_CONTENTSLENGTH
                                   static_cast<const char*>(inputs[ith_input]->DataRaw()));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddFormBuffer(input_names[ith_input].c_str(),
                                   fake_filename.c_str(),
                                   inputs[ith_input]->DataRaw(),
                                   inputs[ith_input]->SizeInBytes());
        break;
      default:
        ORTX_CXX_API_THROW("input must be either text or binary", ORT_INVALID_ARGUMENT);
        break;
    }
  }
}

void OpenAIAudioToTextInvoker::ProcessResponse(const std::string& response, ortc::Variadic& outputs) const {
  auto& string_tensor = outputs.AllocateStringTensor(0);
  string_tensor.SetStringOutput(std::vector<std::string>{response}, std::vector<int64_t>{1});
}
}  // namespace ort_extensions

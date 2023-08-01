// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "openai_invokers.hpp"

namespace ort_extensions {

OpenAIAudioToTextInvoker::OpenAIAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CurlInvoker(api, info) {
  audio_format_ = TryToGetAttributeWithDefault<std::string>(kAudioFormat, "");

  // OpenAI audio endpoints require 'file' and 'model'.
  // 'model' comes from the node attributes so check 'file' is present in the inputs.
  const auto& property_names = PropertyNames();
  bool have_required_input = std::any_of(property_names.begin(), property_names.end(),
                                         [](const auto& name) { return name == "file"; });

  if (!have_required_input) {
    ORTX_CXX_API_THROW("Required 'file' input was not found", ORT_INVALID_ARGUMENT);
  }
}

void OpenAIAudioToTextInvoker::ValidateArgs(const ortc::Variadic& inputs) const {
  // We don't have a way to get the output type from the custom op API.
  // If there's a mismatch it will fail in the Compute when it allocates the output tensor.
  if (OutputNames().size() != 1) {
    ORTX_CXX_API_THROW("Expected single output", ORT_INVALID_ARGUMENT);
  }
}

void OpenAIAudioToTextInvoker::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  // theoretically the filename the content was buffered from
  static const std::string fake_filename = "data_from_input." + audio_format_;

  curl_handler.AddHeader("Content-Type: multipart/form-data");
  curl_handler.AddFormString("model", ModelName().c_str());

  const auto& property_names = PropertyNames();

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddFormString(property_names[ith_input].c_str(),
                                   // assumes null terminated.
                                   // might be safer to pass pointer and length and add use CURLFORM_CONTENTSLENGTH
                                   static_cast<const char*>(inputs[ith_input]->DataRaw()));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddFormBuffer(property_names[ith_input].c_str(),
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

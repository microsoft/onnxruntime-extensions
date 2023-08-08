// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "azure_invokers.hpp"

#include <sstream>

namespace ort_extensions {

////////////////////// AzureAudioToTextInvoker //////////////////////

AzureAudioToTextInvoker::AzureAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CurlInvoker(api, info) {
  audio_format_ = TryToGetAttributeWithDefault<std::string>(kAudioFormat, "");
}

void AzureAudioToTextInvoker::ValidateInputs(const ortc::Variadic& inputs) const {
  // TODO: Validate any required input names are present

  // We don't have a way to get the output type from the custom op API.
  // If there's a mismatch it will fail in the Compute when it allocates the output tensor.
  if (OutputNames().size() != 1) {
    ORTX_CXX_API_THROW("Expected single output", ORT_INVALID_ARGUMENT);
  }
}

void AzureAudioToTextInvoker::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  // theoretically the filename the content was buffered from
  static const std::string fake_filename = "audio." + audio_format_;

  const auto& property_names = RequestPropertyNames();

  curl_handler.AddHeader("Content-Type: multipart/form-data");
  curl_handler.AddFormString("deployment_id", ModelName().c_str());

  // TODO: If the handling here stays the same as in OpenAIAudioToText we can create a helper function to re-use
  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddFormString(property_names[ith_input].c_str(),
                                   static_cast<const char*>(inputs[ith_input]->DataRaw()));  // assumes null terminated
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddFormBuffer(property_names[ith_input].c_str(),
                                   fake_filename.c_str(),
                                   inputs[ith_input]->DataRaw(),
                                   inputs[ith_input]->SizeInBytes());
        break;
      default:
        ORTX_CXX_API_THROW("input must be either text or binary", ORT_RUNTIME_EXCEPTION);
        break;
    }
  }
}

void AzureAudioToTextInvoker::ProcessResponse(const std::string& response, ortc::Variadic& outputs) const {
  auto& string_tensor = outputs.AllocateStringTensor(0);
  string_tensor.SetStringOutput(std::vector<std::string>{response}, std::vector<int64_t>{1});
}

////////////////////// AzureTextToTextInvoker //////////////////////

AzureTextToTextInvoker::AzureTextToTextInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CurlInvoker(api, info) {
}

void AzureTextToTextInvoker::ValidateInputs(const ortc::Variadic& inputs) const {
  if (inputs.Size() != 2 || inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("Expected 2 string inputs of auth_token and text respectively", ORT_INVALID_ARGUMENT);
  }

  // We don't have a way to get the output type from the custom op API.
  // If there's a mismatch it will fail in the Compute when it allocates the output tensor.
  if (OutputNames().size() != 1) {
    ORTX_CXX_API_THROW("Expected single output", ORT_INVALID_ARGUMENT);
  }
}

void AzureTextToTextInvoker::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  gsl::span<const std::string> input_names = InputNames();

  // TODO: assuming we need to create the correct json from the input text
  curl_handler.AddHeader("Content-Type: application/json");

  const auto& text_input = inputs[1];
  curl_handler.SetOption(CURLOPT_POSTFIELDS, text_input->DataRaw());
  curl_handler.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, text_input->SizeInBytes());
}

void AzureTextToTextInvoker::ProcessResponse(const std::string& response, ortc::Variadic& outputs) const {
  auto& string_tensor = outputs.AllocateStringTensor(0);
  string_tensor.SetStringOutput(std::vector<std::string>{response}, std::vector<int64_t>{1});
}

}  // namespace ort_extensions

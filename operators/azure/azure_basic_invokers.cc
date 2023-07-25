// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "azure_basic_invokers.hpp"

#include <sstream>

#include "curl_handler.hpp"

namespace ort_extensions {

////////////////////// AzureCurlInvoker //////////////////////
AzureCurlInvoker::AzureCurlInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : AzureBaseKernel(api, info) {
}

void AzureCurlInvoker::Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) {
  std::string auth_token = GetAuthToken(inputs);

  if (inputs.Size() != InputNames().size()) {
    // TODO: Add something like MakeString from ORT so we can output expected vs actual counts easily
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  if (outputs.Size() != OutputNames().size()) {
    ORTX_CXX_API_THROW("output count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  // do any additional validation of the number and type of inputs/outputs
  ValidateArgs(inputs, outputs);

  // set the options for the curl handler that apply to all usages
  CurlHandler curl_handler(CurlHandler::WriteStringCallback);

  std::string full_auth = std::string{"Authorization: Bearer "} + auth_token;
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.SetOption(CURLOPT_URL, ModelUri().c_str());
  curl_handler.SetOption(CURLOPT_VERBOSE, Verbose());

  std::string response;
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&response);

  SetupRequest(curl_handler, inputs);
  ExecuteRequest(curl_handler);
  ProcessResponse(response, outputs);
}

void AzureCurlInvoker::ExecuteRequest(CurlHandler& curl_handler) const {
  // this is where we could add any logic required to make the request async
  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }
}

////////////////////// AzureAudioToText //////////////////////

AzureAudioToText::AzureAudioToText(const OrtApi& api, const OrtKernelInfo& info)
    : AzureCurlInvoker(api, info) {
  binary_type_ = TryToGetAttributeWithDefault<std::string>(kBinaryType, "");
}

void AzureAudioToText::ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const {
  if (outputs.Size() != 1 || outputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("Expected single string output", ORT_INVALID_ARGUMENT);
  }
}

void AzureAudioToText::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  gsl::span<const std::string> input_names = InputNames();

  curl_handler.AddHeader("Content-Type: multipart/form-data");

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names[ith_input].c_str(),
                             CURLFORM_COPYCONTENTS,
                             inputs[ith_input]->DataRaw(),
                             CURLFORM_END);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names[ith_input].data(),
                             CURLFORM_BUFFER,
                             "non_exist." + binary_type_,
                             CURLFORM_BUFFERPTR,
                             inputs[ith_input]->DataRaw(),
                             CURLFORM_BUFFERLENGTH,
                             inputs[ith_input]->SizeInBytes(),
                             CURLFORM_END);
        break;
      default:
        ORTX_CXX_API_THROW("input must be either text or binary", ORT_RUNTIME_EXCEPTION);
        break;
    }
  }
}

void AzureAudioToText::ProcessResponse(const std::string& response, ortc::Variadic& outputs) const {
  auto& string_tensor = outputs.AllocateStringTensor(0);
  string_tensor.SetStringOutput(std::vector<std::string>{response}, std::vector<int64_t>{1});
}

////////////////////// AzureTextToText //////////////////////

AzureTextToText::AzureTextToText(const OrtApi& api, const OrtKernelInfo& info)
    : AzureCurlInvoker(api, info) {
}

void AzureTextToText::ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const {
  if (inputs.Size() != 2 || inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("Expected 2 string inputs of auth_token and text respectively", ORT_INVALID_ARGUMENT);
  }

  if (outputs.Size() != 1 || outputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("Expected single string output", ORT_INVALID_ARGUMENT);
  }
}

void AzureTextToText::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  gsl::span<const std::string> input_names = InputNames();

  // TODO: assuming we need to create the correct json from the input text
  curl_handler.AddHeader("Content-Type: application/json");

  const auto& text_input = inputs[1];
  curl_handler.SetOption(CURLOPT_POSTFIELDS, text_input->DataRaw());
  curl_handler.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, text_input->SizeInBytes());
}

void AzureTextToText::ProcessResponse(const std::string& response, ortc::Variadic& outputs) const {
  auto& string_tensor = outputs.AllocateStringTensor(0);
  string_tensor.SetStringOutput(std::vector<std::string>{response}, std::vector<int64_t>{1});
}

}  // namespace ort_extensions

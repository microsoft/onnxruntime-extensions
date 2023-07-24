// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "azure_basic_invokers.hpp"

namespace ort_extensions {

AzureAudioInvoker::AzureAudioInvoker(const OrtApi& api, const OrtKernelInfo& info) : AzureInvoker(api, info) {
  binary_type_ = TryToGetAttributeWithDefault<std::string>(kBinaryType, "");
}

void AzureAudioInvoker::Compute(const ortc::Variadic& inputs, ortc::Tensor<std::string>& output) {
  if (inputs.Size() < 1 ||
      inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("invalid inputs, auto token missing", ORT_RUNTIME_EXCEPTION);
  }

  if (inputs.Size() != InputNames().size()) {
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  if (inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING || "auth_token" != input_names_[0]) {
    ORTX_CXX_API_THROW("first input must be a string of auth token", ORT_INVALID_ARGUMENT);
  }

  std::string auth_token = reinterpret_cast<const char*>(inputs[0]->DataRaw());
  std::string full_auth = std::string{"Authorization: Bearer "} + auth_token;

  StringBuffer string_buffer;
  CurlHandler curl_handler(WriteStringCallback);
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.AddHeader("Content-Type: multipart/form-data");

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names_[ith_input].c_str(),
                             CURLFORM_COPYCONTENTS,
                             inputs[ith_input]->DataRaw(),
                             CURLFORM_END);
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        curl_handler.AddForm(CURLFORM_COPYNAME,
                             input_names_[ith_input].data(),
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
  }  // for

  curl_handler.SetOption(CURLOPT_URL, model_uri_.c_str());
  curl_handler.SetOption(CURLOPT_VERBOSE, verbose_);
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&string_buffer);

  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }

  output.SetStringOutput(std::vector<std::string>{string_buffer.ss_.str()}, std::vector<int64_t>{1L});
}

////////////////////// AzureTextInvoker //////////////////////

AzureTextInvoker::AzureTextInvoker(const OrtApi& api, const OrtKernelInfo& info) : AzureInvoker(api, info) {
}

void AzureTextInvoker::Compute(std::string_view auth, std::string_view input, ortc::Tensor<std::string>& output) {
  CurlHandler curl_handler(WriteStringCallback);
  StringBuffer string_buffer;

  std::string full_auth = std::string{"Authorization: Bearer "} + auth.data();
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.AddHeader("Content-Type: application/json");

  curl_handler.SetOption(CURLOPT_URL, model_uri_.c_str());
  curl_handler.SetOption(CURLOPT_POSTFIELDS, input.data());
  curl_handler.SetOption(CURLOPT_POSTFIELDSIZE_LARGE, input.size());
  curl_handler.SetOption(CURLOPT_VERBOSE, verbose_);
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&string_buffer);

  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }

  output.SetStringOutput(std::vector<std::string>{string_buffer.ss_.str()}, std::vector<int64_t>{1L});
}

}  // namespace ort_extensions

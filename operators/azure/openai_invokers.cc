// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "openai_invokers.hpp"

namespace ort_extensions {

OpenAIAudioToTextInvoker::OpenAIAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CurlInvoker(api, info) {
  audio_format_ = TryToGetAttributeWithDefault<std::string>(kAudioFormat, "");

  const auto& property_names = RequestPropertyNames();

  const auto find_optional_input = [&property_names](const std::string& property_name) {
    std::optional<size_t> result;
    auto optional_input = std::find_if(property_names.begin(), property_names.end(),
                                       [&property_name](const auto& name) { return name == property_name; });

    if (optional_input != property_names.end()) {
      result = optional_input - property_names.begin();
    }

    return result;
  };

  filename_input_ = find_optional_input("filename");
  model_name_input_ = find_optional_input("model");

  // OpenAI audio endpoints require 'file' and 'model'.
  if (!std::any_of(property_names.begin(), property_names.end(),
                   [](const auto& name) { return name == "file"; })) {
    ORTX_CXX_API_THROW("Required 'file' input was not found", ORT_INVALID_ARGUMENT);
  }

  if (ModelName().empty() && !model_name_input_) {
    ORTX_CXX_API_THROW("Required 'model' input was not found", ORT_INVALID_ARGUMENT);
  }
}

void OpenAIAudioToTextInvoker::ValidateInputs(const ortc::Variadic& inputs) const {
  // We don't have a way to get the output type from the custom op API.
  // If there's a mismatch it will fail in the Compute when it allocates the output tensor.
  if (OutputNames().size() != 1) {
    ORTX_CXX_API_THROW("Expected single output", ORT_INVALID_ARGUMENT);
  }
}

void OpenAIAudioToTextInvoker::SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const {
  // theoretically the filename the content was buffered from. provides the extensions indicating the audio format
  static const std::string fake_filename = "audio." + audio_format_;

  const auto& property_names = RequestPropertyNames();

  const auto& get_optional_input =
      [&](const std::optional<size_t>& input_idx, const std::string& default_value, size_t min_size = 1) {
        return (input_idx.has_value() && inputs[*input_idx]->SizeInBytes() > min_size)
                   ? static_cast<const char*>(inputs[*input_idx]->DataRaw())
                   : default_value.c_str();
      };

  // filename_input_ is optional in a model. if it's not present, use a fake filename.
  // if it's present make sure it's not a default empty value. as the filename needs to have an extension of
  // mp3, mp4, mpeg, mpga, m4a, wav, or webm it must be at least 4 characters long.
  const char* filename = get_optional_input(filename_input_, fake_filename, 4);

  curl_handler.AddHeader("Content-Type: multipart/form-data");
  // model name could be input or attribute
  curl_handler.AddFormString("model", get_optional_input(model_name_input_, ModelName()));

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    switch (inputs[ith_input]->Type()) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        curl_handler.AddFormString(property_names[ith_input].c_str(),
                                   // assumes null terminated.
                                   // might be safer to pass pointer and length and add use CURLFORM_CONTENTSLENGTH
                                   static_cast<const char*>(inputs[ith_input]->DataRaw()));
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        // only the 'file' input is uint8
        if (property_names[ith_input] != "file") {
          ORTX_CXX_API_THROW("Only the 'file' input should be uint8 data. Invalid input:" + InputNames()[ith_input],
                             ORT_INVALID_ARGUMENT);
        }

        curl_handler.AddFormBuffer(property_names[ith_input].c_str(),
                                   filename,
                                   inputs[ith_input]->DataRaw(),
                                   inputs[ith_input]->SizeInBytes());
        break;
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        // TODO - required to support 'temperature' input.
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

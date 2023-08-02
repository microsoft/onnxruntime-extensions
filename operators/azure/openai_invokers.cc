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

  auto filename_input = std::find_if(property_names.begin(), property_names.end(),
                                     [](const auto& name) { return name == "filename"; });

  // save the index of the 'filename' input
  if (filename_input != property_names.end()) {
    filename_input_ = filename_input - property_names.begin();
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
  // theoretically the filename the content was buffered from. provides the extensions indicating the audio format
  static const std::string fake_filename = "user_audio." + audio_format_;

  curl_handler.AddHeader("Content-Type: multipart/form-data");
  curl_handler.AddFormString("model", ModelName().c_str());

  const auto& property_names = PropertyNames();

  // filename_input_ is optional in a model. if it's not present, use a fake filename.
  // if it's present make sure it's not a default empty value. as the filename needs to have an extension of
  // mp3, mp4, mpeg, mpga, m4a, wav, or webm it must be at least 4 characters long.
  const char* filename = (filename_input_.has_value() && inputs[*filename_input_]->SizeInBytes() > 4)
                             ? static_cast<const char*>(inputs[*filename_input_]->DataRaw())
                             : fake_filename.c_str();

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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ocos.h"
#include "curl_invoker.hpp"

namespace ort_extensions {

////////////////////// OpenAIAudioToTextInvoker //////////////////////

/// <summary>
/// OpenAI Audio to Text
/// Input: auth_token {string}, Request body values {string|uint8} as per https://platform.openai.com/docs/api-reference/audio
/// Output: text {string}
/// </summary>
/// <remarks>
/// 	The URI and `model` input is read from the node attributes and should not be provided as inputs.
/// 	Example input would be:
///       - string tensor named `auth_token` (required, must be first input)
///       - a uint8 tensor named `file` with audio data in the format matching the 'binary_type' attribute (required)
/// 	    - see OpenAI documentation for current supported audio formats
///       - a string tensor named `prompt` (optional)
/// </remarks>
class OpenAIAudioToTextInvoker : public CurlInvoker {
 public:
  OpenAIAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info);

  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) const {
    // use impl from CurlInvoker
    ComputeImpl(inputs, outputs);
  }

 private:
  void ValidateArgs(const ortc::Variadic& inputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;

  // TODO: Consider allowing overriding with an optional input parameter so the audio format is not hardcoded.
  // FWIW the OpenAI Whisper endpoint seems to infer the audio type but requires the filename to use a supported format.
  //   e.g. if the 'file' property name ends in '.wav' the request is processed, and it will happily use mp3 data.
  // Given that it doesn't seem like we need an optional input for now.
  static constexpr const char* kAudioFormat = "audio_format";
  std::string audio_format_;
};

}  // namespace ort_extensions

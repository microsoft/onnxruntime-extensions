// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ocos.h"

#include <optional>

#include "curl_invoker.hpp"

namespace ort_extensions {

////////////////////// OpenAIAudioToTextInvoker //////////////////////

/// <summary>
/// OpenAI Audio to Text
/// Input: auth_token {string}, Request body values {string|uint8} as per https://platform.openai.com/docs/api-reference/audio
/// Output: text {string}
/// </summary>
/// <remarks>
///     The model URI is read from the node attributes.
///     The model name (e.g. 'whisper-1') can be provided as a node attribute or via an input.
///
///     Example input would be:
///       - string tensor named `auth_token` (required, must be first input)
///       - a uint8 tensor named `file` with audio data in the format matching the 'audio_format' attribute (required)
///       - see OpenAI documentation for current supported audio formats
///       - a string tensor named `filename` (optional) with extension indicating the format of the audio data
///         - e.g. 'audio.mp3'
///       - a string tensor named `prompt` (optional)
///
///     NOTE: 'temperature' is not currently supported.
/// </remarks>
class OpenAIAudioToTextInvoker final : public CurlInvoker {
 public:
  OpenAIAudioToTextInvoker(const OrtApi& api, const OrtKernelInfo& info);

  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) const {
    // use impl from CurlInvoker
    ComputeImpl(inputs, outputs);
  }

 private:
  void ValidateInputs(const ortc::Variadic& inputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;

  // audio format to use if the optional 'filename' input is not provided
  static constexpr const char* kAudioFormat = "audio_format";
  std::string audio_format_;
  std::optional<size_t> filename_input_;    // optional override for generated filename using audio_format
  std::optional<size_t> model_name_input_;  // optional override for model_name attribute
};

}  // namespace ort_extensions

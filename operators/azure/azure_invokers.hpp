// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ocos.h"
#include "curl_invoker.hpp"

namespace ort_extensions {

////////////////////// AzureAudioToText //////////////////////

/// <summary>
/// Azure Audio to Text
/// Input: auth_token {string}, ??? (Update when AOAI endpoint is defined)
/// Output: text {string}
/// </summary>
class AzureAudioToText : public CurlInvoker {
 public:
  AzureAudioToText(const OrtApi& api, const OrtKernelInfo& info);

  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) {
    // use impl from CurlInvoker
    ComputeImpl(inputs, outputs);
  }

 private:
  void ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;

  static constexpr const char* kAudioFormat = "binary_type";
  std::string audio_format_;
};

////////////////////// AzureTextToText //////////////////////

/// Azure Text to Text
/// Input: auth_token {string}, text {string}
/// Output: text {string}
struct AzureTextToText : public CurlInvoker {
  AzureTextToText(const OrtApi& api, const OrtKernelInfo& info);
  
  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) {
    // use impl from CurlInvoker
    ComputeImpl(inputs, outputs);
  }

 private:
  void ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;
};

}  // namespace ort_extensions

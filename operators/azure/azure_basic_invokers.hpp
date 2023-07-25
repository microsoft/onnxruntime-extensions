// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ocos.h"
#include "azure_base_kernel.hpp"
#include "curl_handler.hpp"

namespace ort_extensions {

/// <summary>
/// Base class for requests to Azure using Curl
/// </summary>
class AzureCurlInvoker : public AzureBaseKernel {
 protected:
  AzureCurlInvoker(const OrtApi& api, const OrtKernelInfo& info);
  virtual ~AzureCurlInvoker() = default;

 public:
  // Compute method that is used to co-ordinate all Curl based Azure requests
  // TODO: Can this be `const` to enforce it's stateless?
  void Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs);

 private:
  void ExecuteRequest(CurlHandler& handler) const;

  // derived classes can add any arg validation required.
  // input[0] is the auth_token so validation can skip that
  virtual void ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const {}

  // curl_handler has auth token set from input[0].
  virtual void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const = 0;
  virtual void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const = 0;
};

////////////////////// AzureAudioToText //////////////////////

/// <summary>
/// Azure Audio to Text
/// Input: auth_token {string}, audio {string|uint8}   TODO: Not sure how many inputs are expected/required
/// Output: text {string}
/// </summary>
class AzureAudioToText : public AzureCurlInvoker {
 public:
  AzureAudioToText(const OrtApi& api, const OrtKernelInfo& info);

 private:
  void ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;

  static constexpr const char* kBinaryType = "binary_type";  // attribute name for binary type
  std::string binary_type_;
};

////////////////////// AzureTextToText //////////////////////

/// Azure Text to Text
/// Input: auth_token {string}, text {string}
/// Output: text {string}
struct AzureTextToText : public AzureCurlInvoker {
  AzureTextToText(const OrtApi& api, const OrtKernelInfo& info);

 private:
  void ValidateArgs(const ortc::Variadic& inputs, const ortc::Variadic& outputs) const override;
  void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const override;
  void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const override;
};

}  // namespace ort_extensions

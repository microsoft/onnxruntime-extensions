// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "gsl/span"

namespace ort_extensions {

/// <summary>
/// Base kernel for custom ops that call cloud endpoints.
/// </summary>
class CloudBaseKernel : public BaseKernel {
 protected:
  CloudBaseKernel(const OrtApi& api, const OrtKernelInfo& info);
  virtual ~CloudBaseKernel() = default;

  // Names of attributes the custom operator provides.
  static constexpr const char* kUri = "model_uri";           // required
  static constexpr const char* kModelName = "model_name";    // optional
  static constexpr const char* kModelVer = "model_version";  // optional
  static constexpr const char* kVerbose = "verbose";

  static constexpr int kMinimumSupportedOrtVersion = 14;

  const std::string& ModelUri() const { return model_uri_; }
  const std::string& ModelName() const { return model_name_; }
  const std::string& ModelVersion() const { return model_ver_; }
  bool Verbose() const { return verbose_; }

  const gsl::span<const std::string> InputNames() const { return input_names_; }
  const gsl::span<const std::string> OutputNames() const { return output_names_; }

  // Request property names that are parsed from input names. 1:1 with InputNames() values.
  // e.g. 'node0/prompt' -> 'prompt' and that input provides the 'prompt' property in the request to the endpoint.
  // <see cref="GetPropertyNameFromInputName"/> for further details.
  const gsl::span<const std::string> RequestPropertyNames() const { return property_names_; }

  // first input is required to be auth token. validate that and return it.
  std::string GetAuthToken(const ortc::Variadic& inputs) const;

  /// <summary>
  /// Parse the property name to use in the request to the cloud endpoint from a node input name.
  /// Value returned is text following last '/', or the entire string if no '/'.
  ///   e.g. 'node0/prompt' -> 'prompt'
  /// </summary>
  /// <param name="input_name">Node input name.</param>
  /// <returns>Request property name the input is providing data for.</returns>
  static std::string GetPropertyNameFromInputName(const std::string& input_name);

 private:
  std::string model_uri_;
  std::string model_name_;
  std::string model_ver_;
  bool verbose_;

  std::vector<std::string> input_names_;
  std::vector<std::string> property_names_;
  std::vector<std::string> output_names_;
};

}  // namespace ort_extensions

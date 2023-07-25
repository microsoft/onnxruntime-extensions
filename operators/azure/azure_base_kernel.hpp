// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "gsl/span"

namespace ort_extensions {

class AzureBaseKernel : public BaseKernel {
 protected:
  AzureBaseKernel(const OrtApi& api, const OrtKernelInfo& info);
  virtual ~AzureBaseKernel() = default;

  // Names of attributes the custom operator provides.
  static constexpr const char* kUri = "model_uri";
  static constexpr const char* kModelName = "model_name";
  static constexpr const char* kModelVer = "model_version";
  static constexpr const char* kVerbose = "verbose";

  static constexpr int MinimumSupportedOrtVersion = 14;

  const std::string& ModelUri() const { return model_uri_; }
  const std::string& ModelName() const { return model_name_; }
  const std::string& ModelVersion() const { return model_ver_; }
  const std::string& Verbose() const { return verbose_; }

  const gsl::span<const std::string> InputNames() const { return input_names_; }
  const gsl::span<const std::string> OutputNames() const { return output_names_; }

  // first input is required to be auth token. validate that and return it.
  std::string GetAuthToken(const ortc::Variadic& inputs) const;

 private:
  std::string model_uri_;
  std::string model_name_;
  std::string model_ver_;
  std::string verbose_;

  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

}  // namespace ort_extensions

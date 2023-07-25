// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "azure_invoker.hpp"

#include <sstream>

namespace ort_extensions {
AzureBaseKernel::AzureBaseKernel(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  auto ver = GetActiveOrtAPIVersion();
  if (ver < MinimumSupportedOrtVersion) {
    ORTX_CXX_API_THROW("Azure custom operators require onnxruntime version >= 1.14", ORT_RUNTIME_EXCEPTION);
  }

  model_uri_ = TryToGetAttributeWithDefault<std::string>(kUri, "");
  model_name_ = TryToGetAttributeWithDefault<std::string>(kModelName, "");
  model_ver_ = TryToGetAttributeWithDefault<std::string>(kModelVer, "0");
  verbose_ = TryToGetAttributeWithDefault<std::string>(kVerbose, "0");

  OrtStatusPtr status{};
  size_t input_count{};
  status = api_.KernelInfo_GetInputCount(&info_, &input_count);
  if (status) {
    ORTX_CXX_API_THROW("failed to get input count", ORT_RUNTIME_EXCEPTION);
  }

  for (size_t ith_input = 0; ith_input < input_count; ++ith_input) {
    char input_name[1024]{};
    size_t name_size = 1024;
    status = api_.KernelInfo_GetInputName(&info_, ith_input, input_name, &name_size);
    if (status) {
      ORTX_CXX_API_THROW("failed to get name for input " + std::to_string(ith_input), ORT_RUNTIME_EXCEPTION);
    }
    input_names_.push_back(input_name);
  }

  if (input_names_[0] != "auth_token") {
    ORTX_CXX_API_THROW("first input name must be 'auth_token'", ORT_INVALID_ARGUMENT);
  }

  size_t output_count = {};
  status = api_.KernelInfo_GetOutputCount(&info_, &output_count);
  if (status) {
    ORTX_CXX_API_THROW("failed to get output count", ORT_RUNTIME_EXCEPTION);
  }

  for (size_t ith_output = 0; ith_output < output_count; ++ith_output) {
    char output_name[1024]{};
    size_t name_size = 1024;
    status = api_.KernelInfo_GetOutputName(&info_, ith_output, output_name, &name_size);
    if (status) {
      ORTX_CXX_API_THROW("failed to get name for output " + std::to_string(ith_output), ORT_RUNTIME_EXCEPTION);
    }
    output_names_.push_back(output_name);
  }
}

std::string AzureBaseKernel::GetAuthToken(const ortc::Variadic& inputs) const {
  if (inputs.Size() < 1 ||
      inputs[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
    ORTX_CXX_API_THROW("auth_token string is required to be the first input", ORT_INVALID_ARGUMENT);
  }

  std::string auth_token{static_cast<const char*>(inputs[0]->DataRaw())};
}

}  // namespace ort_extensions

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <sstream>
#include "ocos.h"

bool BaseKernel::HasAttribute(const char* name) const noexcept {
  size_t size;
  std::string out;
  // Crashes here.
  OrtStatus* status = api_.KernelInfoGetAttribute_string(&info_, name, nullptr, &size);
  auto r = api_.GetErrorCode(status);
  bool has = (r == ORT_INVALID_ARGUMENT) || (r == ORT_OK);
  if (has) {
    api_.ReleaseStatus(status);
    return has;
  }
  const char* error = api_.GetErrorMessage(status);
  if (strstr(error, "No attribute") == error) {
    api_.ReleaseStatus(status);
    return false;
  }
  api_.ReleaseStatus(status);
  return true;
}

OrtErrorCode BaseKernel::GetErrorCodeAndRelease(OrtStatusPtr status) const noexcept {
  if (status == nullptr) {
    return ORT_OK;
  }
  auto error_code = api_.GetErrorCode(status);
  api_.ReleaseStatus(status);
  return error_code;
}

void BaseKernel::SetOutput(OrtKernelContext* ctx, size_t output_idx, const std::vector<int64_t>& dim,
                           const std::vector<int64_t>& data) {
  OrtValue* output = ort_.KernelContext_GetOutput(ctx, output_idx, dim.data(), dim.size());
  int64_t* data_ptr = ort_.GetTensorMutableData<int64_t>(output);
  for (size_t i = 0; i < data.size(); i++) {
    data_ptr[i] = data[i];
  }
}

template <>
bool BaseKernel::TryToGetAttribute(const char* name, std::string& value) const noexcept {
  size_t size = 0;
  OrtStatus* status = api_.KernelInfoGetAttribute_string(&info_, name, nullptr, &size);

  // The status should be a nullptr when querying for the size.
  if (status != nullptr) {
    api_.ReleaseStatus(status);
    return false;
  }

  value.resize(size);
  status = api_.KernelInfoGetAttribute_string(&info_, name, &value[0], &size);
  if (GetErrorCodeAndRelease(status) != ORT_OK) {
    return false;
  }
  value.resize(size - 1);

  return true;
}

template <>
bool BaseKernel::TryToGetAttribute(const char* name, int64_t& value) const noexcept {
  return GetErrorCodeAndRelease(api_.KernelInfoGetAttribute_int64(&info_, name, &value)) == ORT_OK;
}

template <>
bool BaseKernel::TryToGetAttribute(const char* name, float& value) const noexcept {
  return GetErrorCodeAndRelease(api_.KernelInfoGetAttribute_float(&info_, name, &value)) == ORT_OK;
}

template <>
bool BaseKernel::TryToGetAttribute(const char* name, bool& value) const noexcept {
  int64_t origin_value = 0;
  if (GetErrorCodeAndRelease(api_.KernelInfoGetAttribute_int64(&info_, name, &origin_value)) != ORT_OK) {
    return false;
  }

  value = origin_value == 1;
  return true;
}

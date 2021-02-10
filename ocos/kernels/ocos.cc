// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "utils/string_utils.h"

bool BaseKernel::HasAttribute(const char* name) const {
  if (info_ == nullptr) {
    throw std::runtime_error("Kernel was incorrectly initialized, pointer info_ cannot be null.");
  }
  size_t size;
  std::string out;
  // Crashes here.
  OrtStatus* status = api_.KernelInfoGetAttribute_string(info_, name, nullptr, &size);
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
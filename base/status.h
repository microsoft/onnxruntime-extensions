// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <memory>
#include "ortx_types.h"

struct OrtStatus;

struct OrtxStatus {
  OrtxStatus();
  ~OrtxStatus();
  OrtxStatus(extError_t code, const std::string& error_message);
  OrtxStatus(const OrtxStatus& s);
  OrtxStatus& operator=(const OrtxStatus& s);
  bool operator==(const OrtxStatus& s) const;
  bool operator!=(const OrtxStatus& s) const;
  [[nodiscard]] inline bool IsOk() const { return rep_ == nullptr; }

  void SetErrorMessage(const char* str);
  [[nodiscard]] const char* Message() const;
  [[nodiscard]] extError_t Code() const;

  OrtStatus* CreateOrtStatus() const;

 private:
  struct Rep;
  std::unique_ptr<Rep> rep_;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <memory>

#include "ortx_types.h"

struct OrtStatus;

class OrtxStatus {
  struct Rep {
    extError_t code{kOrtxOK};
    std::string error_message;
  };

 public:
  OrtxStatus() = default;
  ~OrtxStatus() = default;

  OrtxStatus(extError_t code, const std::string& error_message)
      : rep_(std::make_unique<Rep>().release()) {
    rep_->code = code;
    rep_->error_message = std::string(error_message);
  }

  OrtxStatus(const OrtxStatus& s)
      : rep_((s.rep_ == nullptr) ? nullptr : std::make_unique<Rep>(*s.rep_).release()) {}

  OrtxStatus& operator=(const OrtxStatus& s) {
    if (rep_ != s.rep_)
      rep_.reset((s.rep_ == nullptr) ? nullptr : std::make_unique<Rep>(*s.rep_).release());

    return *this;
  }

  bool operator==(const OrtxStatus& s) const { return (rep_ == s.rep_); }
  bool operator!=(const OrtxStatus& s) const { return (rep_ != s.rep_); }
  [[nodiscard]] inline bool IsOk() const noexcept{ return rep_ == nullptr; }

  void SetErrorMessage(const char* str) {
    if (rep_ == nullptr)
      rep_ = std::make_unique<Rep>();
    rep_->error_message = str;
  }

  [[nodiscard]] const char* Message() const noexcept{
    return IsOk() ? "" : rep_->error_message.c_str();
  }

  [[nodiscard]] extError_t Code() const { return IsOk() ? extError_t() : rep_->code; }
  std::string ToString() const {
    if (rep_ == nullptr)
      return "OK";

    std::string result;
    switch (Code()) {
      case extError_t::kOrtxOK:
        result = "Success";
        break;
      case extError_t::kOrtxErrorInvalidArgument:
        result = "Invalid argument";
        break;
      case extError_t::kOrtxErrorOutOfMemory:
        result = "Out of Memory";
        break;
      case extError_t::kOrtxErrorCorruptData:
        result = "Corrupt data";
        break;
      case extError_t::kOrtxErrorInvalidFile:
        result = "Invalid data file";
        break;
      case extError_t::kOrtxErrorNotFound:
        result = "Not found";
        break;
      case extError_t::kOrtxErrorAlreadyExists:
        result = "Already exists";
        break;
      case extError_t::kOrtxErrorOutOfRange:
        result = "Out of range";
        break;
      case extError_t::kOrtxErrorNotImplemented:
        result = "Not implemented";
        break;
      case extError_t::kOrtxErrorInternal:
        result = "Internal";
        break;
      case extError_t::kOrtxErrorUnknown:
        result = "Unknown";
        break;
      default:
        break;
    }

    result += ": ";
    result += rep_->error_message;
    return result;
  }

  operator OrtStatus*() const noexcept;

 private:
  std::unique_ptr<Rep> rep_;
};

#define ORTX_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _status = (expr);         \
    if (_status != nullptr) {      \
      return _status;              \
    }                              \
  } while (0)

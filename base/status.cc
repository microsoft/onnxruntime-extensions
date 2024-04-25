// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "status.h"
#include "ort_c_to_cpp.h"

struct OrtxStatus::Rep {
  extError_t code{kOrtxOK};
  std::string error_message;
};

OrtxStatus::OrtxStatus() = default;
OrtxStatus::~OrtxStatus() = default;

OrtxStatus::OrtxStatus(extError_t code, const std::string& error_message)
    : rep_(new Rep) {
  rep_->code = code;
  rep_->error_message = std::string(error_message);
}

OrtxStatus::OrtxStatus(const OrtxStatus& s)
    : rep_((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_)) {}

OrtxStatus& OrtxStatus::operator=(const OrtxStatus& s) {
  if (rep_ != s.rep_)
    rep_.reset((s.rep_ == nullptr) ? nullptr : new Rep(*s.rep_));

  return *this;
}

bool OrtxStatus::operator==(const OrtxStatus& s) const { return (rep_ == s.rep_); }

bool OrtxStatus::operator!=(const OrtxStatus& s) const { return (rep_ != s.rep_); }

const char* OrtxStatus::Message() const {
  return IsOk() ? "" : rep_->error_message.c_str();
}

void OrtxStatus::SetErrorMessage(const char* str) {
  if (rep_ == nullptr)
    rep_ = std::make_unique<Rep>();
  rep_->error_message = str;
}

extError_t OrtxStatus::Code() const { return IsOk() ? extError_t() : rep_->code; }

OrtStatus* OrtxStatus::CreateOrtStatus() const {
  if (IsOk()) {
    return nullptr;
  }

  OrtStatus* status = OrtW::CreateStatus(Message(), OrtErrorCode::ORT_RUNTIME_EXCEPTION);
  return status;
}

std::string OrtxStatus::ToString() const {
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

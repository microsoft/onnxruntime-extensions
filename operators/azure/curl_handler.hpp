// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

// TODO: Why are we defining CURL_STATICLIB in code vs. in cmake
#define CURL_STATICLIB
#include "curl/curl.h"

namespace ort_extensions {

class CurlHandler {
 public:
  using WriteCallBack = size_t (*)(void*, size_t, size_t, void*);

  CurlHandler(WriteCallBack callback);
  ~CurlHandler() = default;

  static size_t WriteStringCallback(void* contents, size_t size, size_t nmemb, void* userp);

  void AddHeader(const char* data) {
    headers_.reset(curl_slist_append(headers_.release(), data));
  }

  template <typename... Args>
  void AddForm(Args... args) {
    curl_formadd(&from_, &last_, args...);
  }

  template <typename T>
  void SetOption(CURLoption opt, T val) {
    curl_easy_setopt(curl_.get(), opt, val);
  }

  CURLcode Perform() {
    SetOption(CURLOPT_HTTPHEADER, headers_.get());
    if (from_) {
      SetOption(CURLOPT_HTTPPOST, from_);
    }

    return curl_easy_perform(curl_.get());
  }

 private:
  std::unique_ptr<CURL, decltype(curl_easy_cleanup)*> curl_;
  std::unique_ptr<curl_slist, decltype(curl_slist_free_all)*> headers_;
  curl_httppost* from_{};
  curl_httppost* last_{};
  std::unique_ptr<curl_httppost, decltype(curl_formfree)*> from_holder_;  // TODO: Why no last_holder_?
};
}  // namespace ort_extensions

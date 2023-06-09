// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#define CURL_STATICLIB
#include "curl/curl.h"
#include "azure_invoker.hpp"
#include <sstream>

constexpr const char* kUri = "model_uri";
constexpr const char* kModelName = "model_name";
constexpr const char* kVerbose = "verbose";

struct StringBuffer {
  StringBuffer() = default;
  ~StringBuffer() = default;
  std::stringstream ss_;
};

// apply the callback only when response is for sure to be a '/0' terminated string
static size_t WriteStringCallback(void* contents, size_t size, size_t nmemb, void* userp) {
  try {
    size_t realsize = size * nmemb;
    auto buffer = reinterpret_cast<struct StringBuffer*>(userp);
    buffer->ss_.write(reinterpret_cast<const char*>(contents), realsize);
    return realsize;
  } catch (...) {
    // exception caught, abort write
    return CURLcode::CURLE_WRITE_ERROR;
  }
}

using CurlWriteCallBack = size_t (*)(void*, size_t, size_t, void*);

class CurlHandler {
 public:
  CurlHandler(CurlWriteCallBack call_back) : curl_(curl_easy_init(), curl_easy_cleanup),
                                             headers_(nullptr, curl_slist_free_all),
                                             from_holder_(from_, curl_formfree) {
    curl_easy_setopt(curl_.get(), CURLOPT_BUFFERSIZE, 102400L);
    curl_easy_setopt(curl_.get(), CURLOPT_NOPROGRESS, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_USERAGENT, "curl/7.83.1");
    curl_easy_setopt(curl_.get(), CURLOPT_MAXREDIRS, 50L);
    curl_easy_setopt(curl_.get(), CURLOPT_FTP_SKIP_PASV_IP, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl_.get(), CURLOPT_WRITEFUNCTION, call_back);
  }
  ~CurlHandler() = default;

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
    SetOption(CURLOPT_HTTPPOST, from_);
    return curl_easy_perform(curl_.get());
  }

 private:
  std::unique_ptr<CURL, decltype(curl_easy_cleanup)*> curl_;
  std::unique_ptr<curl_slist, decltype(curl_slist_free_all)*> headers_;
  curl_httppost* from_{};
  curl_httppost* last_{};
  std::unique_ptr<curl_httppost, decltype(curl_formfree)*> from_holder_;
};

AzureAudioInvoker::AzureAudioInvoker(const OrtApi& api,
                                     const OrtKernelInfo& info) : BaseKernel(api, info) {
  model_uri_ = TryToGetAttributeWithDefault<std::string>(kUri, "");
  model_name_ = TryToGetAttributeWithDefault<std::string>(kModelName, "");
  verbose_ = TryToGetAttributeWithDefault<bool>(kVerbose, false);
}

void AzureAudioInvoker::Compute(std::string_view auth_token,
                                const ortc::Tensor<int8_t>& audio,
                                ortc::Tensor<std::string>& text) {
  CurlHandler curl_handler(WriteStringCallback);
  StringBuffer string_buffer;

  std::string full_auth = std::string{"Authorization: Bearer "} + auth_token.data();
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.AddHeader("Content-Type: multipart/form-data");

  curl_handler.AddForm(CURLFORM_COPYNAME, "model", CURLFORM_COPYCONTENTS, model_name_.c_str(), CURLFORM_END);
  curl_handler.AddForm(CURLFORM_COPYNAME, "response_format", CURLFORM_COPYCONTENTS, "text", CURLFORM_END);
  curl_handler.AddForm(CURLFORM_COPYNAME, "file", CURLFORM_BUFFER, "non_exist.wav", CURLFORM_BUFFERPTR, audio.Data(),
                       CURLFORM_BUFFERLENGTH, audio.NumberOfElement(), CURLFORM_END);

  curl_handler.SetOption(CURLOPT_URL, model_uri_.c_str());
  curl_handler.SetOption(CURLOPT_VERBOSE, verbose_);
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&string_buffer);

  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    ORTX_CXX_API_THROW(curl_easy_strerror(curl_ret), ORT_FAIL);
  }

  text.SetStringOutput(std::vector<std::string>{string_buffer.ss_.str()}, std::vector<int64_t>{1L});
}

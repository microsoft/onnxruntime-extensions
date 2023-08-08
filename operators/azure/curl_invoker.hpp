// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>

#include "curl/curl.h"

#include "ocos.h"
#include "cloud_base_kernel.hpp"

namespace ort_extensions {

class CurlHandler {
 public:
  using WriteCallBack = size_t (*)(char*, size_t, size_t, void*);

  CurlHandler(WriteCallBack callback);
  ~CurlHandler() = default;

  /// <summary>
  /// Callback to add contents to a string
  /// </summary>
  /// <seealso cref="https://curl.se/libcurl/c/CURLOPT_WRITEFUNCTION.html"/>
  /// <returns>Bytes processed. If this does not match element_size * num_elements the libcurl function
  /// used will return CURLE_WRITE_ERROR</returns>
  static size_t WriteStringCallback(char* contents, size_t element_size, size_t num_elements, void* userdata);

  void AddHeader(const char* data) {
    headers_.reset(curl_slist_append(headers_.release(), data));
  }

  template <typename... Args>
  void AddForm(Args... args) {
    curl_formadd(&from_, &last_, args...);
  }

  void AddFormString(const char* name, const char* value) {
    AddForm(CURLFORM_COPYNAME, name,
            CURLFORM_COPYCONTENTS, value,
            CURLFORM_END);
  }

  void AddFormBuffer(const char* name, const char* buffer_name, const void* buffer_ptr, size_t buffer_len) {
    AddForm(CURLFORM_COPYNAME, name,
            CURLFORM_BUFFER, buffer_name,
            CURLFORM_BUFFERPTR, buffer_ptr,
            CURLFORM_BUFFERLENGTH, buffer_len,
            CURLFORM_END);
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

/// <summary>
/// Base class for requests using Curl
/// </summary>
class CurlInvoker : public CloudBaseKernel {
 protected:
  CurlInvoker(const OrtApi& api, const OrtKernelInfo& info);
  virtual ~CurlInvoker() = default;

  // Compute implementation that is used to co-ordinate all Curl based Azure requests.
  // Derived classes need their own Compute to work with the CustomOpLite infrastructure
  void ComputeImpl(const ortc::Variadic& inputs, ortc::Variadic& outputs) const;

 private:
  void ExecuteRequest(CurlHandler& handler) const;

  // Derived classes can add any arg validation required.
  // Prior to this being called, `inputs` are validated to match the number of input names, and
  // the auth_token has been read from input[0] so validation can skip that.
  //
  // the ortc::Variadic outputs are empty until the Compute populates it, so only output names can be validated
  // and those are available from the base class.
  virtual void ValidateInputs(const ortc::Variadic& inputs) const {}

  // curl_handler has auth token set from input[0].
  virtual void SetupRequest(CurlHandler& curl_handler, const ortc::Variadic& inputs) const = 0;
  virtual void ProcessResponse(const std::string& response, ortc::Variadic& outputs) const = 0;
};
}  // namespace ort_extensions

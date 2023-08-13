// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "curl_invoker.hpp"

#include <iostream>  // TEMP error output
#include <sstream>

namespace ort_extensions {

// apply the callback only when response is for sure to be a '/0' terminated string
size_t CurlHandler::WriteStringCallback(char* contents, size_t element_size, size_t num_elements, void* userdata) {
  try {
    size_t bytes = element_size * num_elements;
    std::string& buffer = *static_cast<std::string*>(userdata);
    buffer.append(contents, bytes);
    return bytes;
  } catch (const std::exception& ex) {
    // TODO: This should be captured/logger properly
    std::cerr << ex.what() << std::endl;
    return 0;
  } catch (...) {
    // exception caught, abort write
    std::cerr << "Unknown exception caught in CurlHandler::WriteStringCallback" << std::endl;
    return 0;
  }
}

CurlHandler::CurlHandler(WriteCallBack callback) : curl_(curl_easy_init(), curl_easy_cleanup),
                                                   headers_(nullptr, curl_slist_free_all),
                                                   from_holder_(from_, curl_formfree) {
  CURL* curl = curl_.get();  // CURL == void* so can't dereference

  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 100 * 1024L);  // how was this size chosen? should it be set on a per operator basis?
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "curl/7.83.1");  // should this value come from the curl src instead of being hardcoded?
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 50L);            // 50 seems like a lot if we're directly calling a specific endpoint
  curl_easy_setopt(curl, CURLOPT_FTP_SKIP_PASV_IP, 1L);      // what does this have to do with http requests?
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);
  
  // TEMP
  // https://stackoverflow.com/questions/25253823/how-to-make-ssl-peer-verify-work-on-android
  // curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);

  // should this be configured via a node attribute? different endpoints may have different timeouts
  curl_easy_setopt(curl, CURLOPT_TIMEOUT, 15);
}

////////////////////// CurlInvoker //////////////////////

CurlInvoker::CurlInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CloudBaseKernel(api, info) {
}

void CurlInvoker::ComputeImpl(const ortc::Variadic& inputs, ortc::Variadic& outputs) const {
  std::string auth_token = GetAuthToken(inputs);

  if (inputs.Size() != InputNames().size()) {
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  // do any additional validation of the number and type of inputs/outputs
  ValidateInputs(inputs);

  // set the options for the curl handler that apply to all usages
  CurlHandler curl_handler(CurlHandler::WriteStringCallback);

  std::string full_auth = std::string{"Authorization: Bearer "} + auth_token;
  curl_handler.AddHeader(full_auth.c_str());
  curl_handler.SetOption(CURLOPT_URL, ModelUri().c_str());
  curl_handler.SetOption(CURLOPT_VERBOSE, Verbose());

  std::string response;
  curl_handler.SetOption(CURLOPT_WRITEDATA, (void*)&response);

  SetupRequest(curl_handler, inputs);
  ExecuteRequest(curl_handler);
  ProcessResponse(response, outputs);
}

void CurlInvoker::ExecuteRequest(CurlHandler& curl_handler) const {
  // this is where we could add any logic required to make the request async or maybe handle retries/cancellation.
  auto curl_ret = curl_handler.Perform();
  if (CURLE_OK != curl_ret) {
    const char* err = curl_easy_strerror(curl_ret);
    KERNEL_LOG(ORT_LOGGING_LEVEL_ERROR, ("Curl error (CURLcode=" + std::to_string(curl_ret) + "): " + err).c_str());
 
    ORTX_CXX_API_THROW(err, ORT_FAIL);
  }
}
}  // namespace ort_extensions

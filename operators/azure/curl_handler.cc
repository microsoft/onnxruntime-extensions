// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "curl_handler.hpp"
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

  curl_easy_setopt(curl, CURLOPT_BUFFERSIZE, 100 * 1024L);  // how was this size chosen?
  curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "curl/7.83.1");  // TODO: should this value come from the curl src?
  curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 50L);            // 50 seems like a lot if we're directly calling a specific endpoint
  curl_easy_setopt(curl, CURLOPT_FTP_SKIP_PASV_IP, 1L);      // what does this have to do with http requests?
  curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, callback);
}

}  // namespace ort_extensions

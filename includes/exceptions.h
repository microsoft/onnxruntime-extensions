// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if defined(__ANDROID__)
#include <android/log.h>
#else
#include <iostream>
#endif

#include <stdexcept>

// FUTURE: We need to do manual init in RegisterCustomOps to use the ORT C++ API
// #ifdef OCOS_SHARED_LIBRARY
//   #define ORT_API_MANUAL_INIT
//   #include "onnxruntime_cxx_api.h"
//   #undef ORT_API_MANUAL_INIT
// #else
//   #include "onnxruntime_cxx_api.h"
// #endif
#include "onnxruntime_c_api.h"

// ORT_FILE is defined in the ORT C API from 1.15 on. Provide simplified definition for older versions.
// On Windows, ORT_FILE is a wchar_t version of the __FILE__ macro.
// Otherwise, ORT_FILE is equivalent to __FILE__.
#ifndef ORT_FILE
#ifdef _WIN32
#define ORT_FILE __FILEW__
#else
#define ORT_FILE __FILE__
#endif
#endif

namespace OrtW {

// All C++ methods that can fail will throw an exception of this type
struct Exception : std::exception {
  Exception(std::string message, OrtErrorCode code) : message_{std::move(message)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

// helper that outputs an error message in a platform aware manner
// Usages:
//  - logging exception message when they may not propagate up
//  - logging failure when using the ORT logger
inline void LogError(const ORTCHAR_T* file, int line, const char* msg) {
#if defined(__ANDROID__)
  __android_log_print(ANDROID_LOG_ERROR, "onnxruntime-extensions", "Error in %s line %d: %s", file, line, msg);
#elif defined(_WIN32)
  // need to use wcerr as ORTCHAR_T is wchar_t on Windows
  std::wcerr << "Error in " << file << " line " << line << ": " << msg << std::endl;
#else
  std::cerr << "Error in " << file << " line " << line << ": " << msg << std::endl;
#endif
}

#ifdef OCOS_NO_EXCEPTIONS
#define ORTX_CXX_API_THROW(msg, code)                                      \
  do {                                                                     \
    OrtW::LogError(ORT_FILE, __LINE__, OrtW::Exception(msg, code).what()); \
    abort();                                                               \
  } while (false)

#define OCOS_TRY if (true)
#define OCOS_CATCH(x) else if (false)
#define OCOS_RETHROW
// In order to ignore the catch statement when a specific exception (not ... ) is caught and referred
// in the body of the catch statements, it is necessary to wrap the body of the catch statement into
// a lambda function. otherwise the exception referred will be undefined and cause build break
#define OCOS_HANDLE_EXCEPTION(func)
#else

// if this is a shared library we need to throw a known exception type as onnxruntime will not know about
// OrtW::Exception.
#ifdef OCOS_SHARED_LIBRARY
#if defined(__ANDROID__)
// onnxruntime and extensions are built with a static libc++ so each has a different definition of
// std::runtime_error, so the ORT output from catching this exception will be 'unknown exception' and the error
// message is lost. log it first so at least it's somewhere
#define ORTX_CXX_API_THROW(msg_in, code)                                   \
  do {                                                                     \
    std::string msg(msg_in);                                               \
    OrtW::LogError(ORT_FILE, __LINE__, msg.c_str());                       \
    throw std::runtime_error((std::to_string(code) + ": " + msg).c_str()); \
  } while (false)
#else
#define ORTX_CXX_API_THROW(msg, code) \
  throw std::runtime_error((std::to_string(code) + ": " + msg).c_str())
#endif
#else
#define ORTX_CXX_API_THROW(msg, code) \
  throw OrtW::Exception(msg, code)
#endif

#define OCOS_TRY try
#define OCOS_CATCH(x) catch (x)
#define OCOS_RETHROW throw;
#define OCOS_HANDLE_EXCEPTION(func) func()
#endif

inline void ThrowOnError(const OrtApi& ort, OrtStatus* status) {
  if (status) {
    std::string error_message = ort.GetErrorMessage(status);
    OrtErrorCode error_code = ort.GetErrorCode(status);
    ort.ReleaseStatus(status);
    ORTX_CXX_API_THROW(std::move(error_message), error_code);
  }
}
}  // namespace OrtW

// macros to wrap entry points that ORT calls where we may need to prevent exceptions propagating upwards to ORT
#define OCOS_API_IMPL_BEGIN \
  OCOS_TRY {
// if exceptions are disabled (a 3rd party library could throw so we need to handle that)
// or we're preventing exception propagation, log and abort().
#if defined(OCOS_NO_EXCEPTIONS) || defined(OCOS_PREVENT_EXCEPTION_PROPAGATION)
#define OCOS_API_IMPL_END                            \
  }                                                  \
  OCOS_CATCH(const std::exception& ex) {             \
    OCOS_HANDLE_EXCEPTION([&]() {                    \
      OrtW::LogError(ORT_FILE, __LINE__, ex.what()); \
      abort();                                       \
    });                                              \
  }
#else
// rethrow.
#define OCOS_API_IMPL_END             \
  }                                   \
  OCOS_CATCH(const std::exception&) { \
    OCOS_HANDLE_EXCEPTION([&]() {     \
      OCOS_RETHROW;                   \
    });                               \
  }
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

// may or may not be available depending on the ORT version
struct OrtLogger;

namespace ort_extensions {
// Disable GCC 'ignoring attributes on template argument' warning due to Logger_LogMessage using
// `__attribute__((warn_unused_result))`.
// The template here is about whether Logger_LogMessage exists or not, so the attribute is irrelevant.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

template <class, class = void>
struct has_Logger_LogMessage : std::false_type {};

// check if Logger_LogMessage exists in OrtApi. Available from 1.15 onwards
template <class T>
struct has_Logger_LogMessage<T, std::void_t<decltype(&T::Logger_LogMessage)>> : std::true_type {};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif

// for use in implementation of classes that have a GetLogger() member that returns an ort_extensions::Logger<T>.
// severity is an ORT_LOGGING_LEVEL_... value (e.g. ORT_LOGGING_LEVEL_WARNING)
#ifdef _WIN32
#define KERNEL_LOG(severity, msg) \
  GetLogger().LogMessage(severity, __FILEW__, __LINE__, __FUNCTION__, msg)
#else
#define KERNEL_LOG(severity, msg) \
  GetLogger().LogMessage(severity, __FILE__, __LINE__, __FUNCTION__, msg)
#endif

struct FallbackLogger {
  static void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                         const char* func_name, const char* message) noexcept {
    // hardcoded when using fallback due to old ORT version.
    if (log_severity_level >= ORT_LOGGING_LEVEL_WARNING) {
      OrtW::LogError(file_path, line_number, (std::string(func_name) + ": " + message).c_str());
    }
  }
};

// implementation for ORT versions < 1.15
template <class T = OrtApi, bool HasLogger = has_Logger_LogMessage<T>::value>
struct LoggerImpl {
  LoggerImpl(const T& api, const OrtKernelInfo& /*info*/) noexcept : api_{api} {
  }

  void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                  const char* func_name, const char* message) const noexcept {
    FallbackLogger::LogMessage(log_severity_level, file_path, line_number, func_name, message);
  }

  const T& api_;
};

// implementation for ORT version >= 1.15 where we can use the ORT logger
template <class T>
struct LoggerImpl<T, true> {
  LoggerImpl(const T& api, const OrtKernelInfo& info) noexcept
      : api_{api}, api_version_{GetActiveOrtAPIVersion()} {
    // Get logger from the OrtKernelInfo should never fail. The logger comes from the EP, and is set when the EP is
    // registered in the InferenceSession, which happens before model load.
    auto status = api_.KernelInfo_GetLogger(&info, &ort_logger_);
    assert(status == nullptr);
  }

  void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                  const char* func_name, const char* message) const noexcept {
    // Logger_LogMessage was added in ORT 1.15
    if (api_version_ >= 15) {
      auto status = api_.Logger_LogMessage(ort_logger_, log_severity_level, message, file_path, line_number, func_name);
      if (status == nullptr) {
        return;
      }

      // Logger_LogMessage shouldn't fail. log why it did and fall through to use DefaultLogMessage.
      OrtW::LogError(file_path, line_number, api_.GetErrorMessage(status));
      api_.ReleaseStatus(status);
    }

    FallbackLogger::LogMessage(log_severity_level, file_path, line_number, func_name, message);
  }

  // T == OrtApi, but we must use T so it compiles when KernelInfo_GetLogger and Logger_LogMessage are not available
  const T& api_;
  const int api_version_;  // runtime API version that RegisterCustomOps was called with
  const OrtLogger* ort_logger_{nullptr};
};

// Logging wrapper to use the ORT logger if available, otherwise fallback to default logging.
class Logger {
 public:
  Logger(const OrtApi& api, const OrtKernelInfo& info) noexcept
      : impl_{api, info} {
  }

  void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                  const char* func_name, const char* message) const noexcept {
    impl_.LogMessage(log_severity_level, file_path, line_number, func_name, message);
  }

 private:
  LoggerImpl<OrtApi> impl_;
};
}  // namespace ort_extensions

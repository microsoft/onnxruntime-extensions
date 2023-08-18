// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

// may or may not be available depending on the ORT version
struct OrtLogger;  // may or may not exist

namespace ort_extensions {
namespace detail {
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

// Logging wrapper to use the ORT logger if available, otherwise fallback to default logging.
template <class T>
class LoggerImpl {
 public:
  LoggerImpl(const T& api, const OrtKernelInfo& info) noexcept
      : api_{api}, api_version_{GetActiveOrtAPIVersion()} {
    if constexpr (has_Logger_LogMessage<T>::value) {
      // Get logger from the OrtKernelInfo should never fail. The logger comes from the EP, and is set when the EP is
      // registered in the InferenceSession, which happens before model load.
      const OrtLogger* logger = nullptr;
      auto status = api.KernelInfo_GetLogger(&info, &logger);
      assert(status == nullptr);
      ort_logger_ = logger;  // save in type agnostic member and static_cast to use.
    }
  }

  void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                  const char* func_name, const char* message) const noexcept {
    if constexpr (has_Logger_LogMessage<T>::value) {
      // Logger_LogMessage was added in ORT 1.15
      if (api_version_ >= 15) {
        auto status = api_.Logger_LogMessage(static_cast<const OrtLogger*>(ort_logger_), log_severity_level, message,
                                             file_path, line_number, func_name);
        if (status == nullptr) {
          return;
        }

        // Logger_LogMessage shouldn't fail. log why it did and fall through to use DefaultLogMessage.
        OrtW::LogError(file_path, line_number, api_.GetErrorMessage(status));
        api_.ReleaseStatus(status);
      }
    }

    // use fallback
    DefaultLogMessage(log_severity_level, file_path, line_number, func_name, message);
  }

  static void DefaultLogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                                const char* func_name, const char* message) noexcept {
    // hardcoded when using fallback due to old ORT version.
    if (log_severity_level >= ORT_LOGGING_LEVEL_WARNING) {
      OrtW::LogError(file_path, line_number, (std::string(func_name) + ": " + message).c_str());
    }
  }

 private:
  // api_ is really OrtApi but we must use T so compilation works when Logger_LogMessage is not available
  const T& api_;
  int api_version_;                       // runtime ORT API version RegisterCustomOps was called with
  const OrtLogger* ort_logger_{nullptr};  // OrtLogger if available
};
}  // namespace detail

// logger is ort_extensions::Logger.
// severity is an ORT_LOGGING_LEVEL_... value (e.g. ORT_LOGGING_LEVEL_WARNING)
#ifdef _WIN32
#define KERNEL_LOG(logger, severity, msg) \
  logger.LogMessage(severity, __FILEW__, __LINE__, __FUNCTION__, msg)
#else
#define KERNEL_LOG(logger, severity, msg) \
  logger.LogMessage(severity, __FILE__, __LINE__, __FUNCTION__, msg)
#endif

using Logger = detail::LoggerImpl<OrtApi>;

}  // namespace ort_extensions

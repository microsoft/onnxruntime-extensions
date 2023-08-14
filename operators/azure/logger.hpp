// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

namespace ort_extensions {

template <class, class = void>
struct has_Logger_LogMessage : std::false_type {};

// check if Logger_LogMessage exists in OrtApi. Available from 1.15 onwards
template <class T>
struct has_Logger_LogMessage<T, std::void_t<decltype(&T::Logger_LogMessage)>> : std::true_type {};

// for use in implementation of classes that have a GetLogger() member that returns an ort_extensions::Logger<T>.
// severity is an ORT_LOGGING_LEVEL_... value (e.g. ORT_LOGGING_LEVEL_WARNING)
#define KERNEL_LOG(severity, msg) \
  GetLogger().LogMessage(severity, ORT_FILE, __LINE__, __FUNCTION__, msg)

// Logging wrapper to use the ORT logger if available, otherwise fallback to default logging.
class Logger {
 public:
  Logger(const OrtApi& api, const OrtKernelInfo& info) noexcept
      : api_{api}, api_version_{GetActiveOrtAPIVersion()} {
    if constexpr (has_Logger_LogMessage<OrtApi>::value) {
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
    if constexpr (has_Logger_LogMessage<OrtApi>::value) {
      // Logger_LogMessage was added in ORT 1.15
      if (api_version_ >= 15) {
        auto status = api_.Logger_LogMessage(static_cast<const OrtLogger*>(ort_logger_), log_severity_level, message,
                                             file_path, line_number, func_name);
        if (status == nullptr) {
          return;
        }

        // Logger_LogMessage shouldn't fail. log why it did and fall through to use DefaultLogMessage.
        OrtW::LogError(ORT_FILE, __LINE__, api_.GetErrorMessage(status));
        api_.ReleaseStatus(status);
      }
    }

    // use fallback
    Logger::DefaultLogMessage(log_severity_level, file_path, line_number, func_name, message);
  }

  static void DefaultLogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                                const char* func_name, const char* message) noexcept {
    // hardcoded when using fallback due to old ORT version.
    if (log_severity_level >= ORT_LOGGING_LEVEL_WARNING) {
      OrtW::LogError(file_path, line_number, (std::string(func_name) + ": " + message).c_str());
    }
  }

 private:
  const OrtApi& api_;
  int api_version_{0};               // ORT API version RegisterCustomOps was called with
  const void* ort_logger_{nullptr};  // OrtLogger if available. Unused otherwise
};
}  // namespace ort_extensions

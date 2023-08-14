// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

namespace ort_extensions {

template <class, class = void>
struct has_Logger_LogMessage : std::false_type {};

template <class T>
struct has_Logger_LogMessage<T, std::void_t<decltype(&T::Logger_LogMessage)>> : std::true_type {};

// for use in implementation of classes that have a GetLogger() member that returns an ort_extensions::Logger<T>.
// severity is an ORT_LOGGING_LEVEL_... value (e.g. ORT_LOGGING_LEVEL_WARNING)
#define KERNEL_LOG(severity, msg) \
  GetLogger().LogMessage(severity, ORT_FILE, __LINE__, __FUNCTION__, msg)

// Logging wrapper to enable use the ORT logger if available, otherwise fallback to default logging.
template <typename T = std::false_type>
class Logger {
 public:
  Logger(const OrtApi& api, const OrtKernelInfo& /*info*/) : api_{api} {
  }

  void LogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                  const char* func_name, const char* message) const {
    DefaultLogMessage(log_severity_level, file_path, line_number, func_name, message);
  };

  static void DefaultLogMessage(OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
                                const char* func_name, const char* message) {
    // hardcoded when using fallback due to old ORT version.
    if (log_severity_level >= ORT_LOGGING_LEVEL_WARNING) {
      OrtW::LogError(file_path, line_number, (std::string(func_name) + ": " + message).c_str());
    }
  }

 private:
  const OrtApi& api_;
  int api_version_{0};  // api version in use
  // OrtLogger if available. Unused otherwise
  const void* ort_logger_{nullptr};
};

// specialization for when ORT logger is available (compiled against and running with ORT version > 1.15
// return 'bool'. the actual type doesn't matter - just has to be different to the default of 'void'
template <>
Logger<has_Logger_LogMessage<OrtApi>::type>::Logger(const OrtApi& api,
                                                    const OrtKernelInfo& info)
    : api_{api}, api_version_{GetActiveOrtAPIVersion()} {
  // Get logger from the OrtKernelInfo should never fail. The logger comes from the EP, and is set when the EP is
  // registered in the InferenceSession, which happens before model load.
  const OrtLogger* logger = nullptr;
  auto status = api.KernelInfo_GetLogger(&info, &logger);
  assert(status == nullptr);
  ort_logger_ = logger;  // save in type agnostic member and static_cast to use.
}

template <>
void Logger<has_Logger_LogMessage<OrtApi>::type>::LogMessage(
    OrtLoggingLevel log_severity_level, const ORTCHAR_T* file_path, int line_number,
    const char* func_name, const char* message) const {
  // Logger_LogMessage was added in ORT 1.15
  if (api_version_ >= 15) {
    auto status = api_.Logger_LogMessage(static_cast<const OrtLogger*>(ort_logger_), log_severity_level, message,
                                         file_path, line_number, func_name);
    if (status == nullptr) {
      return;
    }

    // error logging. should never happen but log that it did and fall through to using DefaultLogMessage
    OrtW::LogError(ORT_FILE, __LINE__, api_.GetErrorMessage(status));
    api_.ReleaseStatus(status);
  }

  // use fallback
  Logger::DefaultLogMessage(log_severity_level, file_path, line_number, func_name, message);
}

}  // namespace ort_extensions
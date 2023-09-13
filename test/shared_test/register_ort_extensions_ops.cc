// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "register_ort_extensions_ops.hpp"

#if defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

static constexpr const char* GetSharedLibraryPath() {
#if defined(_WIN32)
  return "ortextensions.dll";
#elif defined(__APPLE__)
  return "libortextensions.dylib";
#elif defined(__ANDROID__)
  return "libortextensions.so";
#else
  return "lib/libortextensions.so";
#endif
}

void RegisterOrtExtensionsOps(Ort::SessionOptions& session_options) {
  void* handle = nullptr;
  constexpr auto custom_op_library_filename = GetSharedLibraryPath();
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(static_cast<OrtSessionOptions*>(session_options),
                                                           custom_op_library_filename, &handle));
}

#else  // defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

#include "onnxruntime_extensions.h"

void RegisterOrtExtensionsOps(Ort::SessionOptions& session_options) {
  Ort::ThrowOnError(RegisterCustomOps(static_cast<OrtSessionOptions*>(session_options), OrtGetApiBase()));
}

#endif  // defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

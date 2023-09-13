// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "register_ort_extensions_ops.hpp"

#if defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

#if defined(_WIN32)
#include <Windows.h>  // for FreeLibrary()
#else
#include <dlfcn.h>  // for dlclose()
#endif

static void FreeLibraryHandle(void* handle) noexcept {
  if (!handle) {
    return;
  }

#if defined(_WIN32)
  static_cast<void>(::FreeLibrary(reinterpret_cast<HMODULE>(handle)));
#else
  static_cast<void>(::dlclose(handle));
#endif
}

static constexpr const char* GetSharedLibraryPath() {
#if defined(_WIN32)
  return "ortextensions.dll";
#elif defined(__APPLE__)
  return "libortextensions.dylib";
#else
  return "libortextensions.so";
#endif
}

LibraryHandle RegisterOrtExtensionsOps(Ort::SessionOptions& session_options) {
  void* handle = nullptr;
  constexpr auto custom_op_library_filename = GetSharedLibraryPath();
  // TODO upgrade to RegisterCustomOpsLibrary_V2() when the minimum supported ORT version is at least 1.14.
  Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary(static_cast<OrtSessionOptions*>(session_options),
                                                           custom_op_library_filename, &handle));
  return LibraryHandle{handle, &FreeLibraryHandle};
}

#else  // defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

#include "onnxruntime_extensions.h"

LibraryHandle RegisterOrtExtensionsOps(Ort::SessionOptions& session_options) {
  Ort::ThrowOnError(RegisterCustomOps(static_cast<OrtSessionOptions*>(session_options), OrtGetApiBase()));
  return LibraryHandle{nullptr,
                       // deleter does nothing
                       [](void*) {}};
}

#endif  // defined(ORT_EXTENSIONS_UNIT_TEST_USE_EXTENSIONS_SHARED_LIBRARY)

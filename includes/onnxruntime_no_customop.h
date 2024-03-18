// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file defines API which depends on ONNXRuntime, but not including Custom Op and related facilities
// Custom Op and related classes, functions and macros are in onnxruntime_customop.hpp
#pragma once
#include "exceptions.h"

// namespace of ORT ABI Wrapper
namespace OrtW {

class API {
  // To use ONNX C ABI in a way like OrtW::API::CreateStatus.
 public:
  static API& instance(const OrtApi* ort_api = nullptr) noexcept {
    static API self(ort_api);
    return self;
  }

  static OrtStatusPtr CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept {
    return instance()->CreateStatus(code, msg);
  }

  static void ReleaseStatus(OrtStatusPtr ptr) noexcept {
    instance()->ReleaseStatus(ptr);
  }

  template <typename T>
  static OrtStatusPtr KernelInfoGetAttribute(const OrtKernelInfo& info, const char* name, T& value) noexcept;

  static void ThrowOnError(OrtStatusPtr ptr) {
    OrtW::ThrowOnError(instance().api_, ptr);
  }

  // Caller is responsible for releasing OrtMemoryInfo object
  static OrtStatusPtr CreateOrtMemoryInfo(const char* name, enum OrtAllocatorType type, int id, enum OrtMemType mem_type, OrtMemoryInfo** out) noexcept {
    return instance()->CreateMemoryInfo(name, type, id, mem_type, out);
  }
#if ORT_API_VERSION >= 15
  // Caller is responsible for releasing OrtAllocator object: delete static_cast<onnxruntime::OrtAllocatorImpl*> (allocator)
  static OrtStatusPtr GetOrtAllocator(const OrtKernelContext* context, const OrtMemoryInfo* mem_info, OrtAllocator** out) {
    return instance()->KernelContext_GetAllocator(context, mem_info, out);
  }
#endif
 private:
  const OrtApi* operator->() const {
    return &api_;
  }

  API(const OrtApi* api) : api_(*api) {
    if (api == nullptr) {
      ORTX_CXX_API_THROW("ort-extensions internal error: ORT-APIs used before RegisterCustomOps", ORT_RUNTIME_EXCEPTION);
    }
  }

  const OrtApi& api_;
};

template <>
inline OrtStatusPtr API::KernelInfoGetAttribute<int64_t>(const OrtKernelInfo& info, const char* name, int64_t& value) noexcept {
  return instance()->KernelInfoGetAttribute_int64(&info, name, &value);
}

template <>
inline OrtStatusPtr API::KernelInfoGetAttribute<float>(const OrtKernelInfo& info, const char* name, float& value) noexcept {
  return instance()->KernelInfoGetAttribute_float(&info, name, &value);
}

template <>
inline OrtStatusPtr API::KernelInfoGetAttribute<std::string>(const OrtKernelInfo& info, const char* name, std::string& value) noexcept {
  size_t size = 0;
  std::string out;
  // Feed nullptr for the data buffer to query the true size of the string attribute
  OrtStatus* status = instance()->KernelInfoGetAttribute_string(&info, name, nullptr, &size);
  if (status == nullptr) {
    out.resize(size);
    status = instance()->KernelInfoGetAttribute_string(&info, name, &out[0], &size);
    out.resize(size - 1);  // remove the terminating character '\0'
  }

  if (status == nullptr) {
    value = std::move(out);
  }

  return status;
}

template <class T>
inline OrtStatusPtr GetOpAttribute(const OrtKernelInfo& info, const char* name, T& value) noexcept {
  if (auto status = API::KernelInfoGetAttribute(info, name, value); status) {
    // Ideally, we should know which kind of error code can be ignored, but it is not available now.
    // Just ignore all of them.
    API::ReleaseStatus(status);
  }

  return nullptr;
}

template <class T>
inline T GetOpAttributeOrDefault(const OrtKernelInfo& info, const char* name, const T& default_value) noexcept {
  T ret;
  if (API::KernelInfoGetAttribute(info, name, ret)) {
    ret = default_value;
  }
  return ret;
}

inline OrtStatusPtr CreateStatus(const char* msg, OrtErrorCode code) {
  return API::CreateStatus(code, msg);
}

inline OrtStatusPtr CreateStatus(const std::string& msg, OrtErrorCode code) {
  return API::CreateStatus(code, msg.c_str());
}

inline void ReleaseStatus(OrtStatusPtr& status) {
  API::ReleaseStatus(status);
  status = nullptr;
}

}  // namespace OrtW


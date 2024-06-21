// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include "exceptions.h"

// OrtW: ONNX Runtime C ABI Wrapper
namespace OrtW {

struct CustomOpApi {
  CustomOpApi(const OrtApi& api) : api_(api) {}

  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>, float, int64_t, and string
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) const;

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value) const;
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) const;
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) const;
  size_t GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) const;
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values,
                     size_t dim_values_length) const;
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) const;

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value) const;
  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value) const;

  void* GetTensorMutableRawData(_Inout_ OrtValue* value) const;
  const void* GetTensorRawData(_Inout_ const OrtValue* value) const;

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info) const;
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) const;
  size_t KernelContext_GetInputCount(const OrtKernelContext* context) const;
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) const;
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context) const;
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values,
                                    size_t dim_count) const;

  void ThrowOnError(OrtStatus* status) const {
    OrtW::ThrowOnError(api_, status);
  }

  const OrtApi& GetOrtApi() const { return api_; }

 private:
  const OrtApi& api_;
};

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
  static void ReleaseMemoryInfo(OrtMemoryInfo* mem_info) {
    return instance()->ReleaseMemoryInfo(mem_info);
  }
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


//
// Custom OP API Inlines
//

template <>
inline float CustomOpApi::KernelInfoGetAttribute<float>(_In_ const OrtKernelInfo* info, _In_ const char* name) const {
  float out;
  ThrowOnError(api_.KernelInfoGetAttribute_float(info, name, &out));
  return out;
}

template <>
inline int64_t CustomOpApi::KernelInfoGetAttribute<int64_t>(_In_ const OrtKernelInfo* info, _In_ const char* name) const {
  int64_t out;
  ThrowOnError(api_.KernelInfoGetAttribute_int64(info, name, &out));
  return out;
}

template <>
inline std::string CustomOpApi::KernelInfoGetAttribute<std::string>(_In_ const OrtKernelInfo* info, _In_ const char* name) const {
  size_t size = 0;
  std::string out;

  // Feed nullptr for the data buffer to query the true size of the string attribute
  OrtStatus* status = api_.KernelInfoGetAttribute_string(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttribute_string(info, name, &out[0], &size));
    out.resize(size - 1);  // remove the terminating character '\0'
  } else {
    ThrowOnError(status);
  }
  return out;
}

template <>
inline std::vector<float> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) const {
  size_t size = 0;
  std::vector<float> out;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status = api_.KernelInfoGetAttributeArray_float(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttributeArray_float(info, name, out.data(), &size));
  } else {
    ThrowOnError(status);
  }
  return out;
}

template <>
inline std::vector<int64_t> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) const {
  size_t size = 0;
  std::vector<int64_t> out;

  // Feed nullptr for the data buffer to query the true size of the attribute
  OrtStatus* status = api_.KernelInfoGetAttributeArray_int64(info, name, nullptr, &size);

  if (status == nullptr) {
    out.resize(size);
    ThrowOnError(api_.KernelInfoGetAttributeArray_int64(info, name, out.data(), &size));
  } else {
    ThrowOnError(status);
  }
  return out;
}

inline OrtTensorTypeAndShapeInfo* CustomOpApi::GetTensorTypeAndShape(_In_ const OrtValue* value) const {
  OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(api_.GetTensorTypeAndShape(value, &out));
  return out;
}

inline size_t CustomOpApi::GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) const {
  size_t out;
  ThrowOnError(api_.GetTensorShapeElementCount(info, &out));
  return out;
}

inline ONNXTensorElementDataType CustomOpApi::GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) const {
  ONNXTensorElementDataType out;
  ThrowOnError(api_.GetTensorElementType(info, &out));
  return out;
}

inline size_t CustomOpApi::GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) const {
  size_t out;
  ThrowOnError(api_.GetDimensionsCount(info, &out));
  return out;
}

inline void CustomOpApi::GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) const {
  ThrowOnError(api_.GetDimensions(info, dim_values, dim_values_length));
}

inline void CustomOpApi::SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) const {
  ThrowOnError(api_.SetDimensions(info, dim_values, dim_count));
}

template <typename T>
inline T* CustomOpApi::GetTensorMutableData(_Inout_ OrtValue* value) const {
  T* data = nullptr;
  ThrowOnError(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
  return data;
}

template <typename T>
inline const T* CustomOpApi::GetTensorData(_Inout_ const OrtValue* value) const {
  return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
}

inline void* CustomOpApi::GetTensorMutableRawData(_Inout_ OrtValue* value) const {
  void* data = nullptr;
  ThrowOnError(api_.GetTensorMutableData(value, &data));
  return data;
}

inline const void* CustomOpApi::GetTensorRawData(_Inout_ const OrtValue* value) const {
  return GetTensorMutableRawData(const_cast<OrtValue*>(value));
}

inline std::vector<int64_t> CustomOpApi::GetTensorShape(const OrtTensorTypeAndShapeInfo* info) const {
  std::vector<int64_t> output(GetDimensionsCount(info));
  GetDimensions(info, output.data(), output.size());
  return output;
}

inline void CustomOpApi::ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) const {
  api_.ReleaseTensorTypeAndShapeInfo(input);
}

inline size_t CustomOpApi::KernelContext_GetInputCount(const OrtKernelContext* context) const {
  size_t out;
  ThrowOnError(api_.KernelContext_GetInputCount(context, &out));
  return out;
}

inline const OrtValue* CustomOpApi::KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) const {
  const OrtValue* out;
  ThrowOnError(api_.KernelContext_GetInput(context, index, &out));
  return out;
}

inline size_t CustomOpApi::KernelContext_GetOutputCount(const OrtKernelContext* context) const {
  size_t out;
  ThrowOnError(api_.KernelContext_GetOutputCount(context, &out));
  return out;
}

inline OrtValue* CustomOpApi::KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index,
                                                      _In_ const int64_t* dim_values, size_t dim_count) const {
  OrtValue* out;
  ThrowOnError(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, &out));
  return out;
}

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

} // namespace of OrtW


// Deprecated: No needs to create a new class derived from BaseKernel.
struct BaseKernel {
  BaseKernel(const OrtApi& api, const OrtKernelInfo& info) noexcept
      : api_(api), info_(info), ort_(api_) {
  }

  template <class T>
  bool TryToGetAttribute(const char* name, T& value) const noexcept;

  template <class T>
  T TryToGetAttributeWithDefault(const char* name, const T& default_value) const noexcept {
    T result = default_value;
    TryToGetAttribute(name, result);
    return result;
  }

  void SetOutput(OrtKernelContext* ctx, size_t output_idx, const std::vector<int64_t>& dim,
                 const std::vector<int64_t>& data);

 protected:
  OrtErrorCode GetErrorCodeAndRelease(OrtStatusPtr status) const noexcept;

  const OrtApi& api_;
  const OrtKernelInfo& info_;
  OrtW::CustomOpApi ort_;
};

// Deprecated: Use OrtW::CustomOpApi::KernelInfoGetAttribute instead
struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions() = default;
  OrtTensorDimensions(const OrtW::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }

  int64_t Size() const {
    int64_t s = 1;
    for (auto it = begin(); it != end(); ++it)
      s *= *it;
    return s;
  }

  bool IsScalar() const {
    return empty();
  }

  bool IsVector() const {
    return size() == 1;
  }
};

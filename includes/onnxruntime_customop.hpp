// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//.A very thin wrapper of ONNXRuntime Custom Operator Callback ABI, which
// is only used in the custom-op kernels. For the general ORT C++ invocation, like end-to-end
// testing, the ONNXRuntime public C++ APIs should be used since there is no binary compatible requirement.

#pragma once
#include <cstddef>
#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

#ifdef ORT_NO_EXCEPTIONS
#include <iostream>
#endif

#include "onnxruntime_c_api.h"



namespace OrtW {

// All C++ methods that can fail will throw an exception of this type
struct Exception : std::exception {
  Exception(std::string&& string, OrtErrorCode code) : message_{std::move(string)}, code_{code} {}

  OrtErrorCode GetOrtErrorCode() const { return code_; }
  const char* what() const noexcept override { return message_.c_str(); }

 private:
  std::string message_;
  OrtErrorCode code_;
};

#ifdef ORT_NO_EXCEPTIONS
#define ORTX_CXX_API_THROW(string, code)       \
  do {                                        \
    std::cerr << Ort::Exception(string, code) \
                     .what()                  \
              << std::endl;                   \
    abort();                                  \
  } while (false)
#else
#define ORTX_CXX_API_THROW(string, code) \
  throw OrtW::Exception(string, code)
#endif

inline void ThrowOnError(const OrtApi& ort, OrtStatus* status) {
  if (status) {
    std::string error_message = ort.GetErrorMessage(status);
    OrtErrorCode error_code = ort.GetErrorCode(status);
    ort.ReleaseStatus(status);
    ORTX_CXX_API_THROW(std::move(error_message), error_code);
  }
}


//
// Custom OPs (only needed to implement custom OPs)
//
struct CustomOpApi {
  CustomOpApi(const OrtApi& api) : api_(api) {}

  template <typename T>  // T is only implemented for std::vector<float>, std::vector<int64_t>, float, int64_t, and string
  T KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name);

  OrtTensorTypeAndShapeInfo* GetTensorTypeAndShape(_In_ const OrtValue* value);
  size_t GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  ONNXTensorElementDataType GetTensorElementType(const OrtTensorTypeAndShapeInfo* info);
  size_t GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info);
  void GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length);
  void SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count);

  template <typename T>
  T* GetTensorMutableData(_Inout_ OrtValue* value);
  template <typename T>
  const T* GetTensorData(_Inout_ const OrtValue* value);

  std::vector<int64_t> GetTensorShape(const OrtTensorTypeAndShapeInfo* info);
  void ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input);
  size_t KernelContext_GetInputCount(const OrtKernelContext* context);
  const OrtValue* KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index);
  size_t KernelContext_GetOutputCount(const OrtKernelContext* context);
  OrtValue* KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index, _In_ const int64_t* dim_values, size_t dim_count);

  void ThrowOnError(OrtStatus* status) {
    OrtW::ThrowOnError(api_, status);
  }

 private:
  const OrtApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = ORT_API_VERSION;
    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) { return static_cast<const TOp*>(this_)->CreateKernel(*api, info); };
    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetName(); };

    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetExecutionProviderType(); };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetInputTypeCount(); };
    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputType(index); };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) { return static_cast<const TOp*>(this_)->GetOutputTypeCount(); };
    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputType(index); };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) { static_cast<TKernel*>(op_kernel)->Compute(context); };
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetInputCharacteristic(index); };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* this_, size_t index) { return static_cast<const TOp*>(this_)->GetOutputCharacteristic(index); };
  }

  template <typename... Args>
  TKernel* CreateKernelImpl(Args&&... args) const {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
    return new TKernel(std::forward<Args>(args)...);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
  }

  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return CreateKernelImpl(api);
  }

  // Default implementation of GetExecutionProviderType that returns nullptr to default to the CPU provider
  const char* GetExecutionProviderType() const { return nullptr; }

  // Default implementations of GetInputCharacteristic() and GetOutputCharacteristic() below
  // (inputs and outputs are required by default)
  OrtCustomOpInputOutputCharacteristic GetInputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }

  OrtCustomOpInputOutputCharacteristic GetOutputCharacteristic(size_t /*index*/) const {
    return OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED;
  }
};

//
// Custom OP API Inlines
//

template <>
inline float CustomOpApi::KernelInfoGetAttribute<float>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  float out;
  ThrowOnError(api_.KernelInfoGetAttribute_float(info, name, &out));
  return out;
}

template <>
inline int64_t CustomOpApi::KernelInfoGetAttribute<int64_t>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
  int64_t out;
  ThrowOnError(api_.KernelInfoGetAttribute_int64(info, name, &out));
  return out;
}

template <>
inline std::string CustomOpApi::KernelInfoGetAttribute<std::string>(_In_ const OrtKernelInfo* info, _In_ const char* name) {
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
inline std::vector<float> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) {
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
inline std::vector<int64_t> CustomOpApi::KernelInfoGetAttribute(_In_ const OrtKernelInfo* info, _In_ const char* name) {
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

inline OrtTensorTypeAndShapeInfo* CustomOpApi::GetTensorTypeAndShape(_In_ const OrtValue* value) {
  OrtTensorTypeAndShapeInfo* out;
  ThrowOnError(api_.GetTensorTypeAndShape(value, &out));
  return out;
}

inline size_t CustomOpApi::GetTensorShapeElementCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetTensorShapeElementCount(info, &out));
  return out;
}

inline ONNXTensorElementDataType CustomOpApi::GetTensorElementType(const OrtTensorTypeAndShapeInfo* info) {
  ONNXTensorElementDataType out;
  ThrowOnError(api_.GetTensorElementType(info, &out));
  return out;
}

inline size_t CustomOpApi::GetDimensionsCount(_In_ const OrtTensorTypeAndShapeInfo* info) {
  size_t out;
  ThrowOnError(api_.GetDimensionsCount(info, &out));
  return out;
}

inline void CustomOpApi::GetDimensions(_In_ const OrtTensorTypeAndShapeInfo* info, _Out_ int64_t* dim_values, size_t dim_values_length) {
  ThrowOnError(api_.GetDimensions(info, dim_values, dim_values_length));
}

inline void CustomOpApi::SetDimensions(OrtTensorTypeAndShapeInfo* info, _In_ const int64_t* dim_values, size_t dim_count) {
  ThrowOnError(api_.SetDimensions(info, dim_values, dim_count));
}

template <typename T>
inline T* CustomOpApi::GetTensorMutableData(_Inout_ OrtValue* value) {
  T* data;
  ThrowOnError(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
  return data;
}

template <typename T>
inline const T* CustomOpApi::GetTensorData(_Inout_ const OrtValue* value) {
  return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
}

inline std::vector<int64_t> CustomOpApi::GetTensorShape(const OrtTensorTypeAndShapeInfo* info) {
  std::vector<int64_t> output(GetDimensionsCount(info));
  GetDimensions(info, output.data(), output.size());
  return output;
}

inline void CustomOpApi::ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* input) {
  api_.ReleaseTensorTypeAndShapeInfo(input);
}

inline size_t CustomOpApi::KernelContext_GetInputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetInputCount(context, &out));
  return out;
}

inline const OrtValue* CustomOpApi::KernelContext_GetInput(const OrtKernelContext* context, _In_ size_t index) {
  const OrtValue* out;
  ThrowOnError(api_.KernelContext_GetInput(context, index, &out));
  return out;
}

inline size_t CustomOpApi::KernelContext_GetOutputCount(const OrtKernelContext* context) {
  size_t out;
  ThrowOnError(api_.KernelContext_GetOutputCount(context, &out));
  return out;
}

inline OrtValue* CustomOpApi::KernelContext_GetOutput(OrtKernelContext* context, _In_ size_t index,
                                                      _In_ const int64_t* dim_values, size_t dim_count) {
  OrtValue* out;
  ThrowOnError(api_.KernelContext_GetOutput(context, index, dim_values, dim_count, &out));
  return out;
}

}  // namespace OrtW

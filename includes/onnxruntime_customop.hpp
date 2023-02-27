// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//.A very thin wrapper of ONNXRuntime Custom Operator Callback ABI, which
// is only used in the custom-op kernels. For the general ORT C++ invocation, like end-to-end
// testing, the ONNXRuntime public C++ APIs should be used since there is no binary compatible requirement.

#pragma once
#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>

#include "onnxruntime_c_api.h"

#include "exceptions.h"

namespace OrtW {

//
// Custom OPs (only needed to implement custom OPs)
//
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

 private:
  const OrtApi& api_;
};

template <typename TOp, typename TKernel>
struct CustomOpBase : OrtCustomOp {
  CustomOpBase() {
    OrtCustomOp::version = 10;  // The minimum ORT version supported
    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* api, const OrtKernelInfo* info) {
      void* result = nullptr;
      OCOS_API_IMPL_BEGIN
      result = static_cast<const TOp*>(this_)->CreateKernel(*api, *info);
      OCOS_API_IMPL_END
      return result;
    };

    OrtCustomOp::GetName = [](const OrtCustomOp* this_) noexcept {
      return static_cast<const TOp*>(this_)->GetName();
    };

    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) noexcept {
      return static_cast<const TOp*>(this_)->GetExecutionProviderType();
    };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) noexcept {
      return static_cast<const TOp*>(this_)->GetInputTypeCount();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t index) noexcept {
      return static_cast<const TOp*>(this_)->GetInputType(index);
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) noexcept {
      return static_cast<const TOp*>(this_)->GetOutputTypeCount();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t index) noexcept {
      return static_cast<const TOp*>(this_)->GetOutputType(index);
    };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      OCOS_API_IMPL_BEGIN
      static_cast<TKernel*>(op_kernel)->Compute(context);
      OCOS_API_IMPL_END
    };

#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
    OrtCustomOp::KernelDestroy = [](void* op_kernel) { delete static_cast<TKernel*>(op_kernel); };
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* this_, size_t index) noexcept {
      return static_cast<const TOp*>(this_)->GetInputCharacteristic(index);
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* this_, size_t index) noexcept {
      return static_cast<const TOp*>(this_)->GetOutputCharacteristic(index);
    };
  }

  // default implementation. we can't use a virtual function as the layout of this struct has to be aligned with
  // OrtCustomOp, but a derived class can override by creating a function with the same name and signature,
  // calling this base class implementation as needed. e.g. see CustomOpThree in the unit test code
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo& info) const {
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
#pragma warning(disable : 26409)
#endif
    return new TKernel(api, info);
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
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
  T* data;
  ThrowOnError(api_.GetTensorMutableData(value, reinterpret_cast<void**>(&data)));
  return data;
}

template <typename T>
inline const T* CustomOpApi::GetTensorData(_Inout_ const OrtValue* value) const {
  return GetTensorMutableData<T>(const_cast<OrtValue*>(value));
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

}  // namespace OrtW

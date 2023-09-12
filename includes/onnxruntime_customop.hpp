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
#include <optional>

#include "onnxruntime_c_api.h"
#include "exceptions.h"


#define MIN_ORT_VERSION_SUPPORTED 11

extern "C" int ORT_API_CALL GetActiveOrtAPIVersion();

// namespace of ORT ABI Wrapper
namespace OrtW {

class API {
  // To use ONNX C ABI in a way like OrtW::API::CreateStatus.
 public:
  static API& instance(const OrtApi* ort_api = nullptr) noexcept {
    static API self(*ort_api);
    assert(self.api_ != nullptr);
    return self;
  }

  static OrtStatusPtr CreateStatus(OrtErrorCode code, _In_ const char* msg) noexcept {
    return instance()->CreateStatus(code, msg);
  }

  static void ReleaseStatus(OrtStatusPtr ptr) noexcept {
    instance()->ReleaseStatus(ptr);
  }

  template<typename T>
  static OrtStatusPtr KernelInfoGetAttribute(const OrtKernelInfo& info, const char* name, T& value) noexcept;

  template <class T>
  static OrtStatusPtr TryToGetAttribute(const OrtKernelInfo& info, const char* name, T& value) noexcept {
    if (auto status = KernelInfoGetAttribute(info, name, value); status) {
      // Ideally, we should know which kind of error code can be ignored, but it is not availabe now.
      // Just ignore all of them.
      ReleaseStatus(status);
    }

    return nullptr;
  }

private:
  const OrtApi* operator->() const {
    return &api_;
  }

  API(const OrtApi& api) : api_(api) {
    if (&api == nullptr) {
      ORTX_CXX_API_THROW("ort-extensions internal error: ORT-APIs used before RegisterCustomOps", ORT_RUNTIME_EXCEPTION);
    }
  }
  const OrtApi& api_;
};

// Create C++ Status without OrtApi dependency
class StatusMsg : std::optional<std::tuple<std::string, OrtErrorCode>> {
  // StatusMsg doens't own the OrtStatusPtr, it is only used to pass the OrtStatusPtr
  // since OrtStatus Pointer usually is consumed by the caller of the API, which is used to 
  // be ONNX Runtime C API. not any functions in this header.
 public:
  // accept all optional constructors
  StatusMsg(std::optional<std::tuple<std::string, OrtErrorCode>> msg)
      : std::optional<std::tuple<std::string, OrtErrorCode>>(msg) {
    if (this->has_value()) {
      auto& [msg, code] = this->value();
      status_ = API::CreateStatus(code, msg.c_str());
    }
  }

  StatusMsg(OrtStatusPtr status) : status_(status) {}

  bool IsOk() const {
    return status_ == nullptr;
  }

  operator OrtStatusPtr() const {
    return status_;
  }

  OrtStatusPtr ToOrtStatus() const {
    return status_;
  }

 private:
  OrtStatusPtr status_{};
};

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

  const OrtApi& GetOrtApi() const { return api_; }

 private:
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

inline StatusMsg CreateStatusMsg(const char* msg, OrtErrorCode code) {
  return std::make_tuple(msg, code);
}

inline StatusMsg CreateStatusMsg(OrtStatusPtr status) {
  return StatusMsg(status);
}

}  // namespace OrtW


#if ORT_API_VERSION < 15
#include "custom_op_lite.h"

#else
// From onnxruntime 1.17, the custom op lite API header is used the one from onnxruntime package.
// #include "onnxruntime_lite_custom_op.h"
// The existing custom op lite API header has more features than the one from onnxruntime 1.16.
#include "custom_op_lite.h"

#endif // ORT_API_VERSION < 15



namespace Ort {
namespace Custom {


template <typename... Args>
struct FunctionKernel {
  using ComputeFn = std::function<OrtW::StatusMsg(Args...)>;

  OrtW::StatusMsg Compute(Args... args) const {
    return compute_fn_(args...);
  }

  ComputeFn compute_fn_;
};

// primary template handles types that have no nested ::type member:
template <class, class = void>
struct IsFunctionKernel : std::false_type {};

// specialization recognizes types that do have a nested ::type member:
template <class T>
struct IsFunctionKernel<T, std::void_t<typename T::ComputeFn>> : std::true_type {};

// Helper type
template <typename T>
struct ComputeArgsList;

// Specialization for member function
template <typename C, typename... Args>
struct ComputeArgsList<OrtW::StatusMsg (C::*)(Args...) const> {
  using FunctionType = OrtW::StatusMsg (*)(Args...);
  using MemberFunctionType = OrtW::StatusMsg (C::*)(Args...) const;
};

template <typename CustomOpKernel>
struct OrtLiteCustomStructV2 : public OrtLiteCustomOp {
  using ComputeFunction = decltype(&CustomOpKernel::Compute);
  using RegularComputeType = typename ComputeArgsList<ComputeFunction>::FunctionType;

  template <typename... Args>
  using MemberComputeType = OrtW::StatusMsg (CustomOpKernel::*)(Args...) const;

  struct KernelEx : public CustomOpKernel {
    struct {
      std::string ep_{};
      std::unique_ptr<OrtW::CustomOpApi> api_;
    } extra_;
  };

  template <bool b>
  static OrtW::StatusMsg InitKernel(KernelEx& kernel,
                          const OrtApi& api, const OrtKernelInfo& info, RegularComputeType fn = nullptr) {
    return kernel.OnModelAttach(api, info);
  }

  template <>
  static OrtW::StatusMsg InitKernel<true>(
                          KernelEx& kernel,
                          const OrtApi& api, const OrtKernelInfo& info, RegularComputeType fn) {
    kernel.compute_fn_ = fn;
    return std::nullopt;
  }

  template <typename... Args>
  static void InvokeCompute(const KernelEx& kernel, Args&... t_args) {
    auto status = kernel.Compute(t_args...);
    kernel.extra_.api_->ThrowOnError(status.ToOrtStatus());
  }

  template <typename... Args>
  void ParseArgs(MemberComputeType<Args...> fn) {
    OrtLiteCustomOp::ParseArgs<Args...>(OrtLiteCustomOp::input_types_, OrtLiteCustomOp::output_types_);
  }

  // TODO: consider to disable these legacy functions for mobile build to save binary size
  template <typename... Args>
  void DefineCallbackFunctionsLegacy(MemberComputeType<Args...> fn, RegularComputeType regular_fn) {

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort_api, const OrtKernelInfo* info) {
      auto self = static_cast<const OrtLiteCustomStructV2<CustomOpKernel>*>(this_);
      auto kernel = std::make_unique<KernelEx>();
      OrtW::StatusMsg status = InitKernel<IsFunctionKernel<CustomOpKernel>::value>(*kernel, *ort_api, *info, self->regular_fn_);
      OrtW::ThrowOnError(*ort_api, status.ToOrtStatus());

      kernel->extra_.ep_ = self->execution_provider_;
      kernel->extra_.api_ = std::make_unique<OrtW::CustomOpApi>(*ort_api);
      return reinterpret_cast<void*>(kernel.release());
    };

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto kernel = reinterpret_cast<KernelEx*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->extra_.api_.get(),
                                          context,
                                          tensors,
                                          kernel->extra_.api_->KernelContext_GetInputCount(context),
                                          kernel->extra_.api_->KernelContext_GetOutputCount(context),
                                          kernel->extra_.ep_);
      std::apply([kernel](Args const&... t_args) { InvokeCompute(*kernel, t_args...); }, t);
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      std::unique_ptr<KernelEx>(reinterpret_cast<KernelEx*>(op_kernel)).reset();
    };
  }

#if ORT_API_VERSION > 15
  template <typename... Args>
  void DefineCallbackFunctions(MemberComputeType<Args...> fn, RegularComputeType regular_fn) {
    OrtCustomOp::CreateKernel = nullptr;
    OrtCustomOp::KernelCompute = nullptr;

    OrtCustomOp::CreateKernelV2 = [](const OrtCustomOp* this_,
                                     const OrtApi* api, const OrtKernelInfo* info, void** op_kernel) -> OrtStatusPtr {
      if (api == nullptr) {
        assert(false && "Got a null pointer for ORT api on calling CreateKernelV2");
        // should never happened, what we can do?
        return nullptr;
      }

      if (this_ == nullptr || info == nullptr || op_kernel == nullptr) {
        return api->CreateStatus(ORT_INVALID_ARGUMENT, "OrtCustomOp::CreateKernelV2: received a null pointer");
      }

      auto self = static_cast<const OrtLiteCustomStructV2<CustomOpKernel>*>(this_);
      auto kernel = std::make_unique<KernelEx>();
      if (kernel == nullptr) {
        return api->CreateStatus(ORT_FAIL, "OrtCustomOp::CreateKernelV2: failed to new a kernel, OOM?");
      }

      OrtW::StatusMsg status = InitKernel<IsFunctionKernel<CustomOpKernel>::value>(*kernel, *api, *info, self->regular_fn_);
      if (status.IsOk()) {
        kernel->extra_.ep_ = self->execution_provider_;
        kernel->extra_.api_ = std::make_unique<OrtW::CustomOpApi>(*api);
        *op_kernel = reinterpret_cast<void*>(kernel.release());
      }

      return status.ToOrtStatus();
    };

    OrtCustomOp::KernelComputeV2 = [](void* op_kernel, OrtKernelContext* context) -> OrtStatusPtr {
      auto kernel = reinterpret_cast<KernelEx* >(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->extra_.api_.get(),
                                          context,
                                          tensors,
                                          kernel->extra_.api_->KernelContext_GetInputCount(context),
                                          kernel->extra_.api_->KernelContext_GetOutputCount(context),
                                          kernel->extra_.ep_);
      return std::apply([kernel](Args const&... t_args) {
        return kernel->Compute(t_args...).ToOrtStatus(); }, t);
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      std::unique_ptr<KernelEx>(reinterpret_cast<KernelEx*>(op_kernel)).reset();
    };
  }
#endif // ORT_API_VERSION > 15

  OrtLiteCustomStructV2(const char* op_name,
                        const char* execution_provider,
                        RegularComputeType fn_compute = nullptr)
      : OrtLiteCustomOp(op_name, execution_provider) {

    ParseArgs(&CustomOpKernel::Compute);

#if ORT_API_VERSION > 15
    if (OrtCustomOp::version > 15) {
      DefineCallbackFunctions(&CustomOpKernel::Compute, fn_compute);
    } else
#endif  // ORT_API_VERSION > 15

    {
      DefineCallbackFunctionsLegacy(&CustomOpKernel::Compute, fn_compute);
    }
  }

  RegularComputeType regular_fn_{};
};

template <typename... Args>
OrtLiteCustomOp* CreateLiteCustomOpV2(const char* op_name,
                                      const char* execution_provider,
                                      OrtW::StatusMsg (*custom_compute_fn)(Args...)) {
  using LiteOp = OrtLiteCustomStructV2<FunctionKernel<Args...>>;
  return std::make_unique<LiteOp>(op_name, execution_provider, custom_compute_fn).release();
}

template <typename OpKernel>
OrtLiteCustomOp* CreateLiteCustomOpV2(const char* op_name,
                                      const char* execution_provider) {
  using LiteOp = OrtLiteCustomStructV2<OpKernel>;
  return std::make_unique<LiteOp>(op_name, execution_provider).release();
}

}  // namespace Custom
}  // namespace Ort

namespace ortc = Ort::Custom;

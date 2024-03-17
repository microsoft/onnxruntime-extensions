// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//.A very thin wrapper of ONNXRuntime Custom Operator Callback ABI, which
// is only used in the custom-op kernels. For the general ORT C++ invocation, like end-to-end
// testing, the ONNXRuntime public C++ APIs should be used since there is no binary compatible requirement.

#pragma once
#include <cstdint>
#include <cstddef>
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <utility>
#include <type_traits>
#include <optional>
#include <functional>

#include "exceptions.h"
#include "onnxruntime_no_customop.h"
#include "onnxruntime_cpp_api_legacy.hpp"
#include "onnxruntime_extensions.h"
#include "custom_op_lite.h"

#define MIN_ORT_VERSION_SUPPORTED 11

namespace Ort {
namespace Custom {

template <typename... Args>
struct FunctionKernel {
  using ComputeFn = std::function<OrtStatusPtr(Args...)>;

  OrtStatusPtr Compute(Args... args) const {
    return compute_fn_(args...);
  }

  ComputeFn compute_fn_;
};

// primary template handles types that have no nested ::type member:
template <class, class = void>
struct IsFunctionKernel {
  typedef std::false_type type;
};

// specialization recognizes types that do have a nested ::type member:
template <class T>
struct IsFunctionKernel<T, std::void_t<typename T::ComputeFn>> {
  typedef std::true_type type;
};

// Helper type
template <typename T>
struct ComputeArgsList;

// Specialization for member function
template <typename C, typename... Args>
struct ComputeArgsList<OrtStatusPtr (C::*)(Args...) const> {
  using FunctionType = OrtStatusPtr (*)(Args...);
  using MemberFunctionType = OrtStatusPtr (C::*)(Args...) const;
};

template <typename T, typename = void>
struct CustomOp_defined_getInputMemoryType : std::false_type {};

template <typename T>
struct CustomOp_defined_getInputMemoryType<T, std::void_t<decltype(&T::GetInputMemoryType)>> : std::true_type {};

template <typename CustomOpKernel>
struct OrtLiteCustomStructV2 : public OrtLiteCustomOp {
  using ComputeFunction = decltype(&CustomOpKernel::Compute);
  using RegularComputeType = typename ComputeArgsList<ComputeFunction>::FunctionType;

  template <typename... Args>
  using MemberComputeType = OrtStatusPtr (CustomOpKernel::*)(Args...) const;

  struct KernelEx : public CustomOpKernel {
    struct {
      std::string ep_{};
      std::unique_ptr<OrtW::CustomOpApi> api_;
    } extra_;
  };

  template <typename T>
  static OrtStatusPtr InitKernel(KernelEx& kernel,
                                 const OrtApi& api, const OrtKernelInfo& info, RegularComputeType fn, T t) {
    return kernel.OnModelAttach(api, info);
  }

  static OrtStatusPtr InitKernel(
      KernelEx& kernel,
      const OrtApi& api, const OrtKernelInfo& info, RegularComputeType fn, std::true_type) {
    kernel.compute_fn_ = fn;
    return nullptr;
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
      typedef typename IsFunctionKernel<CustomOpKernel>::type type_flag;
      OrtStatusPtr status = InitKernel(*kernel, *ort_api, *info, self->regular_fn_, type_flag());
      OrtW::ThrowOnError(*ort_api, status);

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
      std::apply([kernel](Args const&... t_args) {
        auto status = kernel->Compute(t_args...); OrtW::API::ThrowOnError(status); }, t);
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      std::unique_ptr<KernelEx>(reinterpret_cast<KernelEx*>(op_kernel)).reset();
    };
  }

#if ORT_API_VERSION >= 16
  template <typename... Args>
  void DefineCallbackFunctions(MemberComputeType<Args...> fn, RegularComputeType regular_fn) {
    OrtCustomOp::CreateKernel = nullptr;
    OrtCustomOp::KernelCompute = nullptr;

    if constexpr (CustomOp_defined_getInputMemoryType<CustomOpKernel>::value) {
      OrtCustomOp::GetInputMemoryType = [](const OrtCustomOp* /*this_*/, size_t index) -> OrtMemType {
        return CustomOpKernel::GetInputMemoryType(index);
      };
    }

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

      typedef typename IsFunctionKernel<CustomOpKernel>::type flag_type;
      OrtStatusPtr status = InitKernel(*kernel, *api, *info, self->regular_fn_, flag_type());
      if (status == nullptr) {
        kernel->extra_.ep_ = self->execution_provider_;
        kernel->extra_.api_ = std::make_unique<OrtW::CustomOpApi>(*api);
        *op_kernel = reinterpret_cast<void*>(kernel.release());
      }

      return status;
    };

    OrtCustomOp::KernelComputeV2 = [](void* op_kernel, OrtKernelContext* context) -> OrtStatusPtr {
      auto kernel = reinterpret_cast<KernelEx*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->extra_.api_.get(),
                                          context,
                                          tensors,
                                          kernel->extra_.api_->KernelContext_GetInputCount(context),
                                          kernel->extra_.api_->KernelContext_GetOutputCount(context),
                                          kernel->extra_.ep_);
      return std::apply([kernel](Args const&... t_args) { return kernel->Compute(t_args...); }, t);
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      std::unique_ptr<KernelEx>(reinterpret_cast<KernelEx*>(op_kernel)).reset();
    };
  }
#endif  // ORT_API_VERSION >= 16

  OrtLiteCustomStructV2(const char* op_name,
                        const char* execution_provider,
                        RegularComputeType fn_compute = nullptr)
      : OrtLiteCustomOp(op_name, execution_provider), regular_fn_(fn_compute) {
    ParseArgs(&CustomOpKernel::Compute);

#if ORT_API_VERSION >= 16
    if (OrtCustomOp::version > 15) {
      DefineCallbackFunctions(&CustomOpKernel::Compute, fn_compute);
    } else
#endif  // ORT_API_VERSION >= 16
    {
      DefineCallbackFunctionsLegacy(&CustomOpKernel::Compute, fn_compute);
    }
  }

  RegularComputeType regular_fn_{};
};

template <typename... Args>
OrtLiteCustomOp* CreateLiteCustomOpV2(const char* op_name,
                                      const char* execution_provider,
                                      OrtStatusPtr (*custom_compute_fn)(Args...)) {
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

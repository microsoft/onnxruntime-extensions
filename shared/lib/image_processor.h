// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <tuple>
#include <vector>
#include <unordered_map>

#include "ortx_processor.h"
#include "c_api_utils.hpp"

#include "ocos.h"
#include "status.h"

namespace ort_extensions {

using ImageRawData = std::vector<uint8_t>;
using TensorArgs = std::vector<ortc::TensorBase*>;

class KernelClass {
 public:
  KernelClass() = default;
  virtual ~KernelClass() = default;
  virtual TensorArgs AllocateOutput(ortc::IAllocator* allocator) const = 0;
  virtual OrtxStatus Apply(TensorArgs& inputs, TensorArgs& output) const = 0;

  template <typename... Args>
  using tuple_function_args = std::tuple<typename std::remove_reference<Args>::type*...>;

  template <typename T>
  static const T& CastInputType(TensorArgs& tensors, int n) {
    using TT = std::remove_reference_t<T>;
    return *static_cast<TT*>(tensors[n]);
  }

  template <typename T, typename... FxArgs>
  static auto CastInputType(TensorArgs& tensors, int n) {
    if (!std::is_const<T>::value) {
      return std::make_tuple();
    }

    return std::tuple_cat(std::make_tuple(CastInputType<T>(tensors, n)), CastInputType<FxArgs...>(tensors, n + 1));
  }

  // static auto CastOutputType(TensorArgs::iterator tensor) {
  //   // return std::make_tuple(static_cast<T&>(*tensor));
  //   return std::make_tuple();
  // }

  // template <typename T, typename ECT = void>
  // static auto CastOutputImpl(TensorArgs::iterator tensor) {
  //   return std::make_tuple();
  // }

  // template <typename T, typename std::enable_if<!std::is_const<std::remove_pointer_t<T>>::value>::type>
  // static auto CastOutputImpl(TensorArgs::iterator tensor) {
  //   return std::make_tuple(static_cast<T&>(*tensor));
  // }

  template <typename T>
  static std::tuple<T> CastOutputImpl(TensorArgs::iterator tensor) {
    return std::make_tuple(static_cast<T>(*tensor));
  }

  // template <typename T, typename ECT = void>
  // static auto OutputTensorTuple(ortc::IAllocator* p_allocator, TensorArgs::iterator tensor) {
  //   return std::make_tuple();
  // }

  // template <typename T, typename std::enable_if<!std::is_const<T>::value>::type>
  // static auto OutputTensorTuple(TensorArgs::iterator tensor) {
  //   return std::make_tuple(static_cast<T&>(*tensor));
  // }

  template <typename T, typename EIT = void>
  static ortc::TensorBase* AllocateTensor(ortc::IAllocator* allocator) {
    return nullptr;
  }

  template <typename T, typename std::enable_if_t<std::is_const<T>::value, int> = 0>
  static ortc::TensorBase* AllocateTensor(ortc::IAllocator* allocator) {
    return std::make_unique<T>(allocator).release();
  }

  // template <typename... TensorArgs>
  // static auto AllocateOutputImpl(ortc::IAllocator* allocator) {
  //   return std::make_tuple(AllocateTensor<TensorArgs>(allocator)...);
  // }

  // template <typename Tuple /*,
  //            typename T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>> */
  //           >
  // static auto /*std::vector<T> */ ToVector(Tuple&& tuple) {
  //   return std::apply([](auto&&... elems) {
  //     return std::vector<std::unique_ptr<ortc::TensorBase>>{std::forward<decltype(elems)>(elems)...};}, std::forward<Tuple>(tuple));
  // }

  template <typename... Args>
  static auto AllocateTuple(ortc::IAllocator* allocator, std::tuple<Args...>* = nullptr) {
    return std::make_tuple(AllocateTensor<Args>(allocator)...);
  }

  template <typename... Args>
  static std::vector<ortc::TensorBase*> AllocateOutput(ortc::IAllocator* allocator) {
    using tuple_no_ref = std::tuple<typename std::remove_reference<Args>::type...>;
    auto result = AllocateTuple<tuple_no_ref>(allocator);
    return std::apply([](auto&&... elems) { return std::vector<ortc::TensorBase*>{elems...}; }, std::move(result));
  }

  // template <typename... Args>
  // auto AllocateOutput(ortc::IAllocator* allocator) const {
  //   auto result = AllocateOutputImpl<Args...>(allocator);
  //   auto tuple = std::apply([](auto&&... args) { return std::make_tuple(args.release()...); }, result);
  //   std::tuple<ortc::TensorBase*...> base_tuple = std::apply([](auto&&... args) { return std::make_tuple(static_cast<ortc::TensorBase*>(args)...); }, tuple);
  //   return base_tuple;
  // }

  // template <typename T, typename... FxArgs>
  // static auto CastOutputType(TensorArgs::iterator tensor) {
  //   return std::tuple_cat(CastOutputImpl<T>(tensor), CastOutputType<FxArgs...>(tensor + 1));
  // }

  static auto CastOutputAllType(TensorArgs::iterator tensor) {
    return std::make_tuple();
  }

  template <typename T, typename... Args>
  static auto CastOutputAllType(TensorArgs::iterator tensor, T& arg, Args&... args) {
    // return std::make_tuple(static_cast<T&>(*tensor), CastOutputAllType(args...));
    return std::tuple_cat(CastOutputImpl<T>(tensor),
                          CastOutputAllType(tensor + 1, args...));
  }

  template <typename... Args>
  static auto CastTensors(TensorArgs& tensors) {
    tuple_function_args<Args...> args{};
    return std::apply([&tensors](auto&... args) { return CastOutputAllType(tensors.begin(), args...); }, args);
  }
};

// template <typename... Args>
// using tuple_with_removed_refs = std::tuple<typename std::remove_reference<Args>::type...>;

// template <typename... Args>
// tuple_with_removed_refs<Args...> remove_ref_from_tuple_members(std::tuple<Args...> const& t) {
//     return tuple_with_removed_refs<Args...> { t };
// }

template <typename... Args>
class KernelFunction : public KernelClass {
 public:
  KernelFunction(OrtxStatus (*body)(Args...)) : body_(body){};
  virtual ~KernelFunction() = default;

  OrtxStatus Compute(Args... args) const {
    return body_(args...);
  }

  TensorArgs AllocateOutput(ortc::IAllocator* allocator) const override {
    auto tensors = KernelClass::AllocateOutput<Args...>(allocator);
    TensorArgs all_args;
    for (auto& tensor : tensors) {
      if (tensor == nullptr) {
        return {};
      } else {
        all_args.push_back(tensor);
      }
    }

    return all_args;
    //  auto base_tuple = std::apply([](auto&&... args) { return std::make_tuple(static_cast<ortc::TensorBase*>(args...)); }, tuple);
    // TensorArgs tensors = std::apply([](auto&&... args) { return std::vector<ortc::TensorBase*>{args...}; }, ts_tuple);
    //  TensorArgs tensors = KernelClass::ToVector(ts_tuple);
    // size_t n = sizeof...(Args) - 1;
    // for (; n >= 0; --n) {
    //   if (tensors[n] == nullptr) {
    //     break;
    //   }
    // }

    // return {tensors.begin() + n, tensors.end()};
  }

  OrtxStatus Apply(TensorArgs& inputs, TensorArgs& outputs) const override {
    TensorArgs all_args;
    all_args.reserve(inputs.size() + outputs.size());
    all_args.insert(all_args.end(), inputs.begin(), inputs.end());
    all_args.insert(all_args.end(), outputs.begin(), outputs.end());
    // auto input_output = std::tuple_cat(CastTensors<Args...>(inputs), CastTensors<Args...>(outputs));
    auto args_tuple = std::tuple_cat(CastTensors<Args...>(all_args));
    // auto input_output = CastInputType<Args...>(inputs, 0);
    return std::apply([this](auto&&... args) { return this->Compute(*args...); }, args_tuple);
  }

 private:
  std::function<OrtxStatus(Args...)> body_;
};

// template <typename T>
// class KernelObject : public KernelClass {
//  public:
//   KernelObject(T& kernel_obj) : kernel_obj_(kernel_obj){};
//   virtual ~KernelObject() = default;

//   template <typename... Args>
//   struct ComputeFunction {
//     OrtxStatus operator()(Args... args) {
//       auto input_output = std::tuple_cat(
//           KernelClass::CastInputType<Args>(args...) +
//           KernelClass::CastOutputType<Args>(args...));
//       return kernel_obj_.Compute(args...);
//     }

//     TensorArgs CreateOutput(ortc::IAllocator* allocator) {
//       return TensorArgs();
//     }

//     T& kernel_obj_;
//   };

//   OrtxStatus Apply(TensorArgs& inputs, TensorArgs& output) const override {
//     auto c_function = ComputeFunction<typename T::Compute>(body_);
//     return std::apply([](auto&&... args) { return c_function(std::forward<Args>(args...)); }, args);
//   }

//   TensorArgs AllocateOutput(ortc::IAllocator* allocator) const override {
//     return ComputeFunction<typename T::Compute>::CreateOutput(allocator);
//   }

//  private:
//   T body_;
// };

// template <typename KernelClass>
// std::unique_ptr<KernelClass> DefineKernelObject() {
//   return std::make_unique<KernelObject<KernelClass>>();
// }

template <typename... Args>
std::unique_ptr<KernelClass> DefineKernelFunction(OrtxStatus (*body)(Args...)) {
  return std::make_unique<KernelFunction<Args...>>(body);
}

class Operation {
 public:
  Operation(const KernelClass& knl) : kernel_(knl){};
  static OrtxStatus Create(std::string_view op_def, std::unique_ptr<Operation>& op);
  virtual ~Operation();

  std::tuple<OrtxStatus, std::vector<ortc::TensorBase*>>
  Apply(ortc::IAllocator* allocator, std::vector<ortc::TensorBase*> inputs) {
    auto outputs = kernel_.AllocateOutput(allocator);
    auto status = kernel_.Apply(inputs, outputs);
    return std::make_tuple(status, outputs);
  }

  void ResetTensors(ortc::IAllocator* allocator);

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> outputs_;
  OrtxStatus status_;

  static std::unordered_map<std::string_view, std::function<std::unique_ptr<KernelClass>()>> kernel_registry_;
  const KernelClass& kernel_;

 private:
  std::string op_name_;
  std::unordered_map<std::string, std::string> attributes_;
  ortc::IAllocator* allocator_{};
};

class ImageProcessor : public OrtxObjectImpl {
 public:
  ImageProcessor() : OrtxObjectImpl(kOrtxKindProcessor){};
  virtual ~ImageProcessor() = default;

  OrtxStatus Init(std::string_view processor_def);

  OrtxStatus PreProcess(
      ort_extensions::span<ImageRawData> image_data,
      ortc::Tensor<float>** pixel_values,
      ortc::Tensor<int64_t>** image_sizes,
      ortc::Tensor<int64_t>** num_img_takens);

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  ortc::IAllocator* allocator_;
};

}  // namespace ort_extensions

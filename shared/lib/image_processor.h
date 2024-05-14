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

std::tuple<std::unique_ptr<ImageRawData[]>, size_t>
LoadRawImages(const std::initializer_list<const char*>& image_paths);

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

  template <typename T>
  static std::tuple<T> CastOutputImpl(TensorArgs::iterator tensor) {
    return std::make_tuple(static_cast<T>(*tensor));
  }

  template <typename T>
  static typename std::enable_if<std::is_const<T>::value, ortc::TensorBase*>::type
  AllocateTensor(ortc::IAllocator* allocator) {
    return nullptr;
  }

  template <typename T>
  static typename std::enable_if<!std::is_const<T>::value, ortc::TensorBase*>::type
  AllocateTensor(ortc::IAllocator* allocator) {
    return std::make_unique<T>(allocator).release();
  }

  template <typename... Args>
  static auto AllocateTuple(ortc::IAllocator* allocator, std::tuple<Args...>*) {
    return std::make_tuple(AllocateTensor<Args>(allocator)...);
  }

  template <typename... Args>
  static std::vector<ortc::TensorBase*> AllocateOutput(ortc::IAllocator* allocator) {
    using tuple_no_ref = std::tuple<typename std::remove_reference<Args>::type...>;
    auto result = AllocateTuple(allocator, (tuple_no_ref*)0);
    return std::apply([](auto&&... elems) { return std::vector<ortc::TensorBase*>{std::forward<decltype(elems)>(elems)...}; }, std::move(result));
  }

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

template <typename... Args>
class KernelFunction : public KernelClass {
 public:
  KernelFunction(OrtxStatus (*body)(Args...)) : body_(body){};
  virtual ~KernelFunction() = default;

  OrtxStatus Compute(Args... args) const {
    return body_(std::forward<Args>(args)...);
  }

  TensorArgs AllocateOutput(ortc::IAllocator* allocator) const override {
    auto tensors = KernelClass::AllocateOutput<Args...>(allocator);
    TensorArgs all_args;
    for (auto& tensor : tensors) {
      if (tensor != nullptr) {
        all_args.push_back(tensor);
      }
    }

    return all_args;
  }

  OrtxStatus Apply(TensorArgs& inputs, TensorArgs& outputs) const override {
    TensorArgs all_args;
    all_args.reserve(inputs.size() + outputs.size());
    all_args.insert(all_args.end(), inputs.begin(), inputs.end());
    all_args.insert(all_args.end(), outputs.begin(), outputs.end());
    auto args_tuple = std::tuple_cat(CastTensors<Args...>(all_args));
    return std::apply([this](auto&&... args) { return this->Compute(std::forward<decltype(*args)>(*args)...); }, std::move(args_tuple));
  }

 private:
  std::function<OrtxStatus(Args...)> body_;
};

template <typename... Args>
std::unique_ptr<KernelClass> DefineKernelFunction(OrtxStatus (*body)(Args...)) {
  return std::make_unique<KernelFunction<Args...>>(body);
}

class Operation {
 public:
  Operation() = default;
  OrtxStatus Init(std::string_view op_def);
  virtual ~Operation();

  std::tuple<OrtxStatus, std::vector<ortc::TensorBase*>>
  Apply(ortc::IAllocator* allocator, std::vector<ortc::TensorBase*> inputs) {
    auto outputs = kernel_->AllocateOutput(allocator);
    auto status = kernel_->Apply(inputs, outputs);
    return std::make_tuple(status, outputs);
  }

  void ResetTensors(ortc::IAllocator* allocator);

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> outputs_;

 private:
  static std::unordered_map<std::string_view, std::function<std::unique_ptr<KernelClass>()>> kernel_registry_;

  std::unique_ptr<KernelClass> kernel_;
  std::string op_name_;
  std::unordered_map<std::string, std::string> attributes_;
  ortc::IAllocator* allocator_{};
};

struct ProcessorResult : public OrtxObjectImpl {
  ProcessorResult() : OrtxObjectImpl(kOrtxKindProcessorResult) {}
  ortc::Tensor<float>* pixel_values{};
  ortc::Tensor<int64_t>* image_sizes{};
  ortc::Tensor<int64_t>* num_img_takens{};
};

class ImageProcessor : public OrtxObjectImpl {
 public:
  ImageProcessor();
  virtual ~ImageProcessor() = default;

  OrtxStatus Init(std::string_view processor_def);

  std::tuple<OrtxStatus, ProcessorResult>
  PreProcess(
      ort_extensions::span<ImageRawData> image_data,
      ortc::Tensor<float>** pixel_values,
      ortc::Tensor<int64_t>** image_sizes,
      ortc::Tensor<int64_t>** num_img_takens);

  void ClearOutputs(ProcessorResult* r);

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  ortc::IAllocator* allocator_;
};

}  // namespace ort_extensions

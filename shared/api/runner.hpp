// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <tuple>
#include <string>
#include <vector>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "op_def_struct.h"
#include "c_api_utils.hpp"

namespace ort_extensions {

using json = nlohmann::json;
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
  using KernelRegistry = std::unordered_map<std::string_view, std::function<std::unique_ptr<KernelClass>()>>;
  Operation(const KernelRegistry& registry) { kernel_registry_ = &registry; };

  OrtxStatus Init(std::string_view op_def) {
    // parse the op_def by json
    auto full_json = json::parse(op_def);
    if (!full_json.is_object()) {
      return {kOrtxErrorInvalidArgument, "[Operation]: failed to parse op_def."};
    }

    auto op_json = full_json.at("operation");

    auto op_name = op_json.at("name").get<std::string>();
    if (op_name.empty()) {
      return {kOrtxErrorInvalidArgument, "[Operation]: name field is missing."};
    }

    auto op_type = op_json.at("type").get<std::string>();
    if (op_type.empty()) {
      return {kOrtxErrorInvalidArgument, "[Operation]: type field is missing."};
    }

    auto kernel_iter = kernel_registry_->find(op_type);
    if (kernel_iter == kernel_registry_->end()) {
      return {kOrtxErrorInvalidArgument, "[Operation]: type is not supported."};
    }

    op_name_ = op_name;
    kernel_ = kernel_iter->second();

    /* TODO: parse the attributes
      if (op_json.contains("attrs")) {
        auto attrs = op_json.at("attrs");
        attrs.get_to(attributes_);
      }
    */
    return {};
  }

  virtual ~Operation() {
    ResetTensors(allocator_);
  }

  std::tuple<OrtxStatus, std::vector<ortc::TensorBase*>>
  Apply(ortc::IAllocator* allocator, std::vector<ortc::TensorBase*> inputs) {
    auto outputs = kernel_->AllocateOutput(allocator);
    auto status = kernel_->Apply(inputs, outputs);
    return std::make_tuple(status, outputs);
  }

  void ResetTensors(ortc::IAllocator* allocator) {
    outputs_.clear();
  }

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> outputs_;

 private:
  const KernelRegistry* kernel_registry_;

  std::unique_ptr<KernelClass> kernel_;
  std::string op_name_;
  std::unordered_map<std::string, std::string> attributes_;
  ortc::IAllocator* allocator_{};
};

class OrtxRunner {
 public:
  OrtxRunner(ortc::IAllocator* allocator, Operation** ops, size_t op_num)
      : allocator_(allocator), ops_(ops, ops + op_num) {}

  template <typename IT, typename OT>  // batch input/output container
  OrtxStatus Run(IT& input_seq, OT& output_seq) {
    for (size_t i = 0; i < input_seq.size(); ++i) {
      auto& input = *(input_seq.begin() + i);
      Operation* last_op = nullptr;
      // sequentially apply the operations
      for (auto& op : ops_) {
        if (last_op != nullptr) {
          last_op->ResetTensors(allocator_);
        }
        auto [status, ts_output] = op->Apply(allocator_, input);
        if (status.IsOk()) {
          if (op == ops_.back()) {
            output_seq.push_back(ts_output);
          } else {
            input = ts_output;
          }
        } else {
          return status;
        }

        last_op = op;
      }
    }

    return {};
  }

 private:
  std::vector<Operation*> ops_;
  ortc::IAllocator* allocator_;
};

}  // namespace ort_extensions

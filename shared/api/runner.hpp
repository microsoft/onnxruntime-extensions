// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <tuple>
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>

#include "nlohmann/json.hpp"
#include "op_def_struct.h"
#include "c_api_utils.hpp"

namespace ort_extensions {

using json = nlohmann::json;
using TensorPtr = std::unique_ptr<ortc::TensorBase>;
using TensorArgs = std::vector<ortc::TensorBase*>;

class KernelDef {
 public:
  KernelDef() = default;
  virtual ~KernelDef() = default;
  virtual OrtxStatus Init(std::string_view attr) { return {}; }  // no need to be initialized for a kernel function
  virtual TensorArgs AllocateOutput(ortc::IAllocator* allocator) const = 0;
  virtual OrtxStatus Apply(TensorArgs& inputs, TensorArgs& output) const = 0;

  using AttrType =
      std::variant<std::string, double, int64_t, std::vector<std::string>, std::vector<double>, std::vector<int64_t>>;
  using AttrDict = std::unordered_map<std::string, AttrType>;

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
  static typename std::enable_if<std::is_const<T>::value, ortc::TensorBase*>::type AllocateTensor(
      ortc::IAllocator* allocator) {
    return nullptr;
  }

  template <typename T>
  static typename std::enable_if<!std::is_const<T>::value, ortc::TensorBase*>::type AllocateTensor(
      ortc::IAllocator* allocator) {
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
    return std::apply(
        [](auto&&... elems) { return std::vector<ortc::TensorBase*>{std::forward<decltype(elems)>(elems)...}; },
        std::move(result));
  }

  static auto CastOutputAllType(TensorArgs::iterator tensor) { return std::make_tuple(); }

  template <typename T, typename... Args>
  static auto CastOutputAllType(TensorArgs::iterator tensor, T& arg, Args&... args) {
    // return std::make_tuple(static_cast<T&>(*tensor), CastOutputAllType(args...));
    return std::tuple_cat(CastOutputImpl<T>(tensor), CastOutputAllType(tensor + 1, args...));
  }

  template <typename... Args>
  static auto CastTensors(TensorArgs& tensors) {
    tuple_function_args<Args...> args{};
    return std::apply([&tensors](auto&... args) { return CastOutputAllType(tensors.begin(), args...); }, args);
  }
};

template <typename... Args>
class KernelFunction : public KernelDef {
 public:
  KernelFunction(OrtxStatus (*body)(Args...)) : body_(body) {};
  virtual ~KernelFunction() = default;

  TensorArgs AllocateOutput(ortc::IAllocator* allocator) const override {
    auto tensors = KernelDef::AllocateOutput<Args...>(allocator);
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
    return std::apply([this](auto&&... args) { return this->Compute(std::forward<decltype(*args)>(*args)...); },
                      std::move(args_tuple));
  }

 private:
  std::function<OrtxStatus(Args...)> body_;

  OrtxStatus Compute(Args... args) const { return body_(std::forward<Args>(args)...); }
};

template <typename T, typename... Args>
class KernelStruct : public KernelDef {
 public:
  KernelStruct(OrtxStatus (T::*body)(Args...)) : body_(body) {};
  virtual ~KernelStruct() = default;

  TensorArgs AllocateOutput(ortc::IAllocator* allocator) const override {
    auto tensors = KernelDef::AllocateOutput<Args...>(allocator);
    TensorArgs all_args;
    for (auto& tensor : tensors) {
      if (tensor != nullptr) {
        all_args.push_back(tensor);
      }
    }

    return all_args;
  }

  OrtxStatus Init(std::string_view attr_str) override {
    instance_ = std::make_unique<T>();

    AttrDict attr_dict;
    if (attr_str.empty()) {
      return instance_->Init(attr_dict);
    }

    auto attr = json::parse(attr_str, nullptr, false);
    if (attr.is_discarded()) {
      return {kOrtxErrorCorruptData, "Failed to parse JSON for kernel attributes."};
    }
    attr_dict.reserve(attr.size());
    for (auto& [key, value] : attr.items()) {
      if (value.is_string()) {
        attr_dict[key] = value.template get<std::string>();
      } else if (value.is_number_integer() || value.is_number_unsigned()) {
        attr_dict[key] = value.template get<int64_t>();
      } else if (value.is_number_float()) {
        attr_dict[key] = value.template get<double>();
      } else if (value.is_array() && value.size() > 0) {
        auto& elem_0 = value.at(0);
        if (elem_0.is_number_float()) {
          attr_dict[key] = value.template get<std::vector<double>>();
        } else if (elem_0.is_string()) {
          attr_dict[key] = value.template get<std::vector<std::string>>();
        } else if (elem_0.is_number_integer() || elem_0.is_number_unsigned()) {
          attr_dict[key] = value.template get<std::vector<int64_t>>();
        } else {
          return {kOrtxErrorCorruptData, "Unsupported mix types in attribute value."};
        }

      } else {
        return {kOrtxErrorCorruptData, "Invalid attribute type."};
      }
    }

    return instance_->Init(attr_dict);
  }

  OrtxStatus Apply(TensorArgs& inputs, TensorArgs& outputs) const override {
    TensorArgs all_args;
    all_args.reserve(inputs.size() + outputs.size());
    all_args.insert(all_args.end(), inputs.begin(), inputs.end());
    all_args.insert(all_args.end(), outputs.begin(), outputs.end());
    auto args_tuple = std::tuple_cat(CastTensors<Args...>(all_args));
    return std::apply(
        [this](auto&&... args) { return (instance_.get()->*body_)(std::forward<decltype(*args)>(*args)...); },
        std::move(args_tuple));
  }

 private:
  OrtxStatus (T::*body_)(Args...){};
  std::unique_ptr<T> instance_;
};

template <typename... Args>
std::unique_ptr<KernelDef> CreateKernelInstance(OrtxStatus (*body)(Args...)) {
  return std::make_unique<KernelFunction<Args...>>(body);
}

template <typename T, typename... Args>
std::unique_ptr<KernelDef> CreateKernelInstance(OrtxStatus (T::*method)(Args...)) {
  return std::make_unique<KernelStruct<T, Args...>>(method);
}

template <typename T, typename... Args>
std::unique_ptr<KernelDef> CreateKernelInstance(OrtxStatus (T::*method)(Args...) const) {
  return std::make_unique<KernelStruct<T, Args...>>(reinterpret_cast<OrtxStatus (T::*)(Args...)>(method));
}

class Operation {
 public:
  using KernelRegistry = std::unordered_map<std::string_view, std::function<std::unique_ptr<KernelDef>()>>;
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

    std::string attr_str;
    if (op_json.contains("attrs")) {
      auto attrs = op_json.at("attrs");
      attr_str = attrs.dump();
    }

    return kernel_->Init(attr_str);
  }

  virtual ~Operation() { ResetTensors(allocator_); }

  std::tuple<OrtxStatus, std::vector<ortc::TensorBase*>> Apply(ortc::IAllocator* allocator,
                                                               std::vector<ortc::TensorBase*> inputs) {
    auto outputs = kernel_->AllocateOutput(allocator);
    auto status = kernel_->Apply(inputs, outputs);
    return std::make_tuple(status, outputs);
  }

  std::vector<TensorPtr> AllocateOutputs(ortc::IAllocator* allocator) {
    auto tensors = kernel_->AllocateOutput(allocator);
    std::vector<TensorPtr> outputs;
    for (auto& tensor : tensors) {
      outputs.push_back(std::unique_ptr<ortc::TensorBase>(tensor));
    }

    return outputs;
  }

  void ResetTensors(ortc::IAllocator* allocator) { outputs_.clear(); }

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> outputs_;

 private:
  const KernelRegistry* kernel_registry_;

  std::unique_ptr<KernelDef> kernel_;
  std::string op_name_;
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

  static OrtxStatus StackTensors(const std::vector<TensorArgs>& arg_lists, std::vector<TensorPtr>& outputs,
                                 ortc::IAllocator* allocator) {
    if (arg_lists.empty()) {
      return {};
    }

    size_t batch_size = arg_lists.size();
    size_t num_outputs = arg_lists[0].size();
    for (size_t axis = 0; axis < num_outputs; ++axis) {
      std::vector<ortc::TensorBase*> ts_ptrs;
      ts_ptrs.reserve(arg_lists.size());
      std::vector<int64_t> shape = arg_lists[0][axis]->Shape();
      for (auto& ts : arg_lists) {
        if (shape != ts[axis]->Shape()) {
          return {kOrtxErrorInvalidArgument, "[StackTensors]: shapes of tensors to stack are not the same."};
        }
        ts_ptrs.push_back(ts[axis]);
      }

      std::vector<int64_t> output_shape = shape;
      output_shape.insert(output_shape.begin(), batch_size);
      std::byte* tensor_buf = outputs[axis]->AllocateRaw(output_shape);
      for (size_t i = 0; i < batch_size; ++i) {
        auto ts = ts_ptrs[i];
        const std::byte* ts_buff = reinterpret_cast<const std::byte*>(ts->DataRaw());
        auto ts_size = ts->SizeInBytes();
        std::memcpy(tensor_buf + i * ts_size, ts_buff, ts_size);
      }
    }

    return {};
  }

 private:
  ortc::IAllocator* allocator_;
  std::vector<Operation*> ops_;
};

}  // namespace ort_extensions

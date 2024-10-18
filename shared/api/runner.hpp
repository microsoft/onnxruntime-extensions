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
  bool IsSequenceOnly() { return false; }

 private:
  std::vector<std::unique_ptr<ortc::TensorBase>> outputs_;
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
    size_t i = 0;
    Operation* last_op = nullptr;
    for (; i < input_seq.size(); ++i) {
      auto& input = *(input_seq.begin() + i);
      // sequentially apply the operations
      for (auto& op : ops_) {
        if (last_op != nullptr) {
          last_op->ResetTensors(allocator_);
        }
        auto [status, ts_output] = op->Apply(allocator_, input);
        if (status.IsOk()) {
          if (op == ops_.back()) {
            output_seq.push_back(std::move(ts_output));
          } else {
            input = ts_output;
          }
        } else {
          return status;
        }

        if (op->IsSequenceOnly()) {
          break;
        }

        last_op = op;
      }
    }

    if (last_op != nullptr) {
      last_op->ResetTensors(allocator_);
    }

    return {};
  }

  static bool IsGreaterShape(const std::vector<int64_t>& lhs, const std::vector<int64_t>& rhs) {
    if (lhs.size() != rhs.size()) {
      return lhs.size() > rhs.size();
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
      if (lhs[i] != rhs[i]) {
        return lhs[i] > rhs[i];
      }
    }

    return false;
  }

  static void CopyOrPadTensor(const std::vector<int64_t>::const_iterator dest_shape_begin,
                              const std::vector<int64_t>::const_iterator dest_shape_end,
                              const std::vector<int64_t>::const_iterator src_shape_begin,
                              const std::vector<int64_t>::const_iterator src_shape_end,
                              std::byte* dest, const std::byte* src, size_t element_size) {
    // no broadcasting here
    assert(dest_shape_begin != dest_shape_end && src_shape_begin != src_shape_end);
    assert(dest_shape_end - dest_shape_begin == src_shape_end - src_shape_begin);

    if ((dest_shape_begin + 1) == dest_shape_end) {
      std::memcpy(dest, src, element_size * (*src_shape_begin));
      if (*dest_shape_begin > *src_shape_begin) {
        std::memset(dest + *src_shape_begin * element_size, 0, (*dest_shape_begin - *src_shape_begin) * element_size);
      }
      return;
    }

    int64_t dest_chunk_size = 1;
    int64_t src_chunk_size = 1;
    for (auto iter = dest_shape_begin + 1; iter != dest_shape_end; ++iter) {
      dest_chunk_size *= *iter;
    }

    for (auto iter = src_shape_begin + 1; iter != src_shape_end; ++iter) {
      src_chunk_size *= *iter;
    }

    for (int64_t i = 0; i < *dest_shape_begin; ++i) {
      if (i < *src_shape_begin) {
        if (dest_chunk_size == src_chunk_size) {
          std::memcpy(dest + i * dest_chunk_size * element_size, src + i * src_chunk_size * element_size,
                      dest_chunk_size * element_size);
        } else {
          CopyOrPadTensor(dest_shape_begin + 1, dest_shape_end, src_shape_begin + 1, src_shape_end,
                          dest + i * dest_chunk_size * element_size, src + i * src_chunk_size * element_size,
                          element_size);
        }
      } else {
        memset(dest + i * dest_chunk_size * element_size, 0, dest_chunk_size * element_size);
      }
    }
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
      size_t element_size = arg_lists[0][axis]->SizeInBytes() / arg_lists[0][axis]->NumberOfElement();
      bool is_same_shape = true;
      for (auto& ts : arg_lists) {
        if (shape != ts[axis]->Shape()) {
          is_same_shape = false;
          auto dtype = ts[axis]->Type();
          if (dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 && dtype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            return {kOrtxErrorInvalidArgument, "[StackTensors]: shapes of tensors to stack are not the same."};
          }
          if (IsGreaterShape(ts[axis]->Shape(), shape)) {
            shape = ts[axis]->Shape();
          }
        }
        ts_ptrs.push_back(ts[axis]);
      }

      std::vector<int64_t> output_shape = shape;
      // if (!is_same_shape) {
      //   if (ts_ptrs.front()->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      //     return {kOrtxErrorInvalidArgument, "[StackTensors]: shapes of tensors to stack are not the same."};
      //   } else {
      //     // if the shape is not the same, but the type is int64, let's pad the shape before the stack
      //     // since shape is already is the max shape, we don't need to do anything here
      //     ;
      //   }
      // }

      output_shape.insert(output_shape.begin(), batch_size);
      std::byte* tensor_buf = outputs[axis]->AllocateRaw(output_shape);
      auto ts_size = outputs[axis]->SizeInBytes() / batch_size;
      for (size_t i = 0; i < batch_size; ++i) {
        auto ts = ts_ptrs[i];
        const std::byte* ts_buff = reinterpret_cast<const std::byte*>(ts->DataRaw());
        if (is_same_shape /* || ts->Shape() == std::vector<int64_t>(output_shape.begin() + 1, output_shape.end()) */) {
          std::memcpy(tensor_buf + i * ts_size, ts_buff, ts_size);
        } else {
          CopyOrPadTensor(output_shape.begin() + 1, output_shape.end(), ts->Shape().begin(), ts->Shape().end(),
                          tensor_buf + i * ts_size, reinterpret_cast<const std::byte*>(ts->DataRaw()), element_size);
        }
      }
    }

    return {};
  }

 private:
  ortc::IAllocator* allocator_;
  std::vector<Operation*> ops_;
};

}  // namespace ort_extensions

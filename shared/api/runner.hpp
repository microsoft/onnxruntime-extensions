// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <list>
#include <tuple>
#include <string>
#include <vector>
#include <memory>
#include <variant>
#include <type_traits>
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
  virtual int64_t GetOutputCount() const = 0;
  virtual TensorArgs AllocateOutput(ortc::IAllocator* allocator) const = 0;
  virtual OrtxStatus Invoke(TensorArgs& inputs, TensorArgs& output) const = 0;

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

  template <typename T, typename... Args>
  static int64_t CountOutputArgs() {
    if constexpr (sizeof...(Args) == 0) {
      return std::is_const<T>::value ? 0 : 1;
    } else {
      return (std::is_const<T>::value ? 0 : 1) + CountOutputArgs<Args...>();
    }
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

  int64_t GetOutputCount() const override { return CountOutputArgs<Args...>(); }

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

  OrtxStatus Invoke(TensorArgs& inputs, TensorArgs& outputs) const override {
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

  int64_t GetOutputCount() const override { return CountOutputArgs<Args...>(); }

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

  OrtxStatus Invoke(TensorArgs& inputs, TensorArgs& outputs) const override {
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

class TensorLookupTable {
 public:
  using TensorBase = ortc::TensorBase;
  TensorLookupTable() = default;
  ~TensorLookupTable() = default;

  void AddTensorRef(std::string_view name) { tensor_map_.insert({std::string(name), nullptr}); }

  void AddTensor(std::string_view name, std::unique_ptr<TensorBase>&& tensor) {
    auto it = tensor_map_.find(std::string(name));
    if (it == tensor_map_.end()) {
      tensor_map_.emplace(std::string(name), std::move(tensor));
    } else {
      it->second = std::move(tensor);
    }
  }

  ortc::TensorBase* GetTensor(const std::string& name) const {
    auto iter = tensor_map_.find(name);
    if (iter == tensor_map_.end()) {
      return nullptr;
    }

    return iter->second.get();
  }

  bool IsReferenced(const std::string& name) const { return tensor_map_.find(name) != tensor_map_.end(); }

  TensorPtr ReleaseTensor(const std::string& name) {
    if (auto it = tensor_map_.find(name); it != tensor_map_.end()) {
      auto ptr = std::move(it->second);
      tensor_map_.erase(it);
      return ptr;
    }

    return {};
  }

  void Reset() {
    for (auto& [name, tensor] : tensor_map_) {
      tensor.reset();
    }
  }

 private:
  std::unordered_map<std::string, std::unique_ptr<ortc::TensorBase>> tensor_map_;
};

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

    auto inputs_iter = op_json.find("inputs");
    if (inputs_iter != op_json.end()) {
      inputs_spec_.clear();
      for (auto& input : *inputs_iter) {
        auto name = input.get<std::string>();
        inputs_spec_.push_back(std::move(name));
      }
    }

    std::string attr_str;
    if (op_json.contains("attrs")) {
      auto attrs = op_json.at("attrs");
      attr_str = attrs.dump();
    }

    return kernel_->Init(attr_str);
  }

  virtual ~Operation() {}

  OrtxStatus Apply(std::vector<ortc::TensorBase*>& inputs, std::vector<ortc::TensorBase*>& outputs) const {
    return kernel_->Invoke(inputs, outputs);
  }

  std::vector<TensorPtr> AllocateOutputs(ortc::IAllocator* allocator) const {
    auto tensors = kernel_->AllocateOutput(allocator);
    std::vector<TensorPtr> outputs;
    for (auto& tensor : tensors) {
      outputs.push_back(std::unique_ptr<ortc::TensorBase>(tensor));
    }

    // for (size_t i = 0; i < tensors.size(); ++i) {
    //   std::string name = op_name_ + ":" + std::to_string(i);
    //   if (tensor_lookup_table_->GetTensor(name) != nullptr) {
    //     tensor_lookup_table_->AddTensor(name, outputs[i]);
    //   }
    // }

    return outputs;
  }

  auto& GetOpName() const { return op_name_; }
  int64_t GetOutputCount() const { return kernel_->GetOutputCount(); }
  auto& GetInputSpec() const { return inputs_spec_; }

 private:
  const KernelRegistry* kernel_registry_;

  std::string op_name_;
  std::unique_ptr<KernelDef> kernel_;
  std::vector<std::string> inputs_spec_{":*"};
};

class ExecutionPlan {
 public:
  ExecutionPlan() = default;
  OrtxStatus Init(const json& plan, const Operation::KernelRegistry& kernel_registry) {
    for (auto mod_iter = plan.begin(); mod_iter != plan.end(); ++mod_iter) {
      auto op = std::make_unique<Operation>(kernel_registry);
      auto status = op->Init(mod_iter->dump());
      if (!status.IsOk()) {
        return status;
      }

      operations_.push_back(std::move(op));
    }

    return {};
  }

  OrtxStatus PrepareInput(const Operation& op, std::vector<TensorPtr>& ts_output, TensorArgs& ts_inputs,
                          TensorLookupTable& ts_lookup_table) const {
    ts_inputs.clear();

    auto& input_spec = op.GetInputSpec();
    for (auto& spec : input_spec) {
      if (spec == ":*") {
        for (auto& out : ts_output) {
          ts_inputs.push_back(out.get());
        }
        continue;
      } else if (spec[0] == ':' && spec.size() > 1) {
        size_t num = std::strtoul(spec.c_str() + 1, nullptr, 10);
        if (num >= ts_output.size()) {
          return {kOrtxErrorInvalidArgument, "Invalid input index."};
        }
        ts_inputs.push_back(ts_output[num].get());
      } else if (auto ts = ts_lookup_table.GetTensor(spec); ts != nullptr) {
        ts_inputs.push_back(ts);
      } else {
        return {kOrtxErrorInvalidArgument, "Input tensor is unknown: " + spec};
      }
    }

    return {};
  }

  OrtxStatus Excute(ortc::IAllocator* allocator, TensorArgs& input, TensorLookupTable& ts_lookup_table) const {
    for (auto& op : operations_) {
      // add tensor references
      auto spec = op->GetInputSpec();
      for (auto& name : spec) {
        if (!name.empty() && name[0] != ':') {
          ts_lookup_table.AddTensorRef(name);
        }
      }
    }

    TensorArgs ts_input{input.begin(), input.end()};
    // Add the outputs of the last operation to the tensor lookup table
    auto& last_op = operations_.back();
    for (int64_t i = 0; i < last_op->GetOutputCount(); ++i) {
      std::string name = last_op->GetOpName() + ":" + std::to_string(i);
      ts_lookup_table.AddTensorRef(name);
    }

    std::vector<TensorPtr> ts_disposables;
    // sequentially apply the operations
    for (size_t n = 0; n < operations_.size(); ++n) {
      auto& op = operations_[n];
      auto ts_output = op->AllocateOutputs(allocator);
      TensorArgs out_ptrs;
      out_ptrs.reserve(ts_output.size());
      std::transform(ts_output.begin(), ts_output.end(), std::back_inserter(out_ptrs),
                     [](auto& ts) { return ts.get(); });
      auto status = op->Apply(ts_input, out_ptrs);

      for (auto& ts : ts_disposables) {
        ts.reset();
      }
      ts_disposables.clear();

      if (status.IsOk()) {
        if (n < operations_.size() - 1) {
          status = PrepareInput(*operations_[n + 1], ts_output, ts_input, ts_lookup_table);
        }

        size_t i = 0;
        for (size_t i = 0; i < ts_output.size(); i++) {
          auto& out_tensor = ts_output[i];
          std::string tensor_name = op->GetOpName() + ":" + std::to_string(i);
          if (ts_lookup_table.IsReferenced(tensor_name)) {
            ts_lookup_table.AddTensor(tensor_name, std::move(out_tensor));
          } else {
            ts_disposables.push_back(std::move(out_tensor));
          }
        }
      }

      if (!status.IsOk()) {
        return status;
      }
    }

    for (auto& ts : ts_disposables) {
      ts.reset();
    }

    return {};
  }

  TensorArgs RetrieveOutput(TensorLookupTable& ts_lookup_table) const {
    std::vector<ortc::TensorBase*> outputs;
    auto& last_op = operations_.back();
    for (int64_t i = 0; i < last_op->GetOutputCount(); ++i) {
      std::string name = last_op->GetOpName() + ":" + std::to_string(i);
      auto ts = ts_lookup_table.ReleaseTensor(name);
      if (ts != nullptr) {
        outputs.push_back(ts.release());
      }
    }

    return outputs;
  }

  std::vector<TensorPtr> AllocateOutputs(ortc::IAllocator* allocator) const {
    auto& last_op = operations_.back();
    return last_op->AllocateOutputs(allocator);
  }

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
};

class OrtxRunner {
 public:
  OrtxRunner(const ExecutionPlan& plan) : allocator_(&CppAllocator::Instance()), plan_(plan) {}

  OrtxStatus Run(std::vector<TensorArgs>& input_seq, std::vector<TensorArgs>& output_seq) {
    for (size_t i = 0; i < input_seq.size(); ++i) {
      auto& input = *(input_seq.begin() + i);
      auto status = plan_.Excute(allocator_, input, tensor_lookup_table_);
      if (!status.IsOk()) {
        return status;
      }

      output_seq.push_back(plan_.RetrieveOutput(tensor_lookup_table_));
    }

    return {};
  }

  void Release() { tensor_lookup_table_.Reset(); }

  ortc::IAllocator* GetAllocator() const { return allocator_; }

  // template <typename IT, typename OT>  // batch input/output container
  // OrtxStatus Run(IT& input_seq, OT& output_seq) {
  //   size_t i = 0;
  //   Operation* last_op = nullptr;
  //   for (; i < input_seq.size(); ++i) {
  //     auto& input = *(input_seq.begin() + i);
  //     // sequentially apply the operations
  //     for (auto& op : ops_) {
  //       if (last_op != nullptr) {
  //         last_op->ResetTensors(allocator_);
  //       }
  //       auto [status, ts_output] = op->Apply(allocator_, input);
  //       if (status.IsOk()) {
  //         if (op == ops_.back()) {
  //           output_seq.push_back(std::move(ts_output));
  //         } else {
  //           input = ts_output;
  //         }
  //       } else {
  //         return status;
  //       }

  //       last_op = op;
  //     }
  //   }

  //   if (last_op != nullptr) {
  //     last_op->ResetTensors(allocator_);
  //   }

  //   return {};
  // }

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
                              const std::vector<int64_t>::const_iterator src_shape_end, std::byte* dest,
                              const std::byte* src, size_t element_size) {
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
        std::memset(dest + i * dest_chunk_size * element_size, 0, dest_chunk_size * element_size);
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
  const ExecutionPlan& plan_;
  TensorLookupTable tensor_lookup_table_;
};

}  // namespace ort_extensions

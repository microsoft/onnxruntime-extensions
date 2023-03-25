#pragma once
#include "onnxruntime_customop.hpp"

namespace Ort {
namespace Custom2 {

class Tensor {
 public:
  Tensor(OrtKernelContext* ctx) : ctx_(ctx) {}

 protected:
  OrtKernelContext* ctx_;
};

template <typename T>
struct Span {
  const T* data_ = {};
  size_t size_ = {};
  void Assign(const T* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  T operator[](size_t indice) const {
    return data_[indice];
  }
  const T* Data() const { return data_; }
};

template <typename T>
class TensorT : public Tensor {
 public:
  using TT = typename std::remove_reference<T>::type;
  TensorT(const OrtW::CustomOpApi& ort_api, OrtKernelContext* ctx, size_t indice, bool is_input) : Tensor(ctx), indice_(indice), is_input_(is_input), ort_api_(ort_api) {
    if (is_input) {
      const_value_ = ort_api.KernelContext_GetInput(ctx_, indice);
      OrtTensorTypeAndShapeInfo* info = ort_api.GetTensorTypeAndShape(const_value_);
      shape_ = ort_api.GetTensorShape(info);
      ort_api.ReleaseTensorTypeAndShapeInfo(info);
    }
  }
  const std::vector<int64_t>& Shape() const {
    return shape_;
  }
  const TT* Data() const {
    return ort_api_.GetTensorData<TT>(const_value_);
  }
  TT* Allocate(const std::vector<int64_t>& shape) {
    if (!data_) {
      OrtValue* out = ort_api_.KernelContext_GetOutput(ctx_, indice_, shape.data(), shape.size());
      data_ = ort_api_.GetTensorMutableData<TT>(out);
    }
    return data_;
  }
  static TT GetT() { return (TT)0; }

  const Span<T>& AsSpan() {
    // assert shape_ is 1-d
    span_.Assign(Data(), shape_[0]);
    return span_;
  }

  const T& AsScalar() {
    // assert shape_ is {1}
    return *Data();
  }

  int64_t NumerOfElement() const {
    int64_t count = 1;
    for (auto i : shape_)
      count *= i;
    return count;
  }

 private:
  size_t indice_;
  bool is_input_;
  const OrtValue* const_value_;  // for input
  TT* data_{};                   // for output
  std::vector<int64_t> shape_;
  Span<T> span_;
  const OrtW::CustomOpApi& ort_api_;
};

using TensorPtr = std::unique_ptr<Custom2::Tensor>;

template <typename CustomType, typename... Args>
struct OrtCustomOpT2 : public OrtCustomOp {
  using InitFn = CustomType* (*)(const OrtKernelInfo*);
  using ComputeFn = void (*)(Args...);
  using ExitFn = void (*)(CustomType*);
  using MyType = OrtCustomOpT2<CustomType, Args...>;

  OrtCustomOpT2(const char* op_name,
                const char* execution_provider,
                InitFn init_fn,
                ComputeFn compute_fn,
                ExitFn exit_fn) : op_name_(op_name),
                                  execution_provider_(execution_provider),
                                  init_fn_(init_fn),
                                  compute_fn_(compute_fn),
                                  exit_fn_(exit_fn) {
    ParseArgs<Args...>();

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto self = reinterpret_cast<MyType*>(op_kernel);
      if (!self->ort_api_) {
        ORTX_CXX_API_THROW("ort api is not set.", ORT_FAIL);
      }
      OrtW::CustomOpApi ort_api(*self->ort_api_);
      auto t = self->CreateInputTupleInvoker(ort_api, context);
      std::apply([self](Args const&... t_args) { self->compute_fn_(t_args...); }, t);
    };

    OrtCustomOp::version = ORT_API_VERSION;

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort, const OrtKernelInfo* info) {
      auto self = const_cast<MyType*>(reinterpret_cast<const MyType*>(this_));
      if (self->init_fn_) {
        self->custom_handle_ = self->init_fn_(info);
      }
      self->ort_api_ = ort;
      return (void*)this_;
    };

    OrtCustomOp::GetName = [](const OrtCustomOp* this_) { return static_cast<const MyType*>(this_)->op_name_.c_str(); };
    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* this_) { return ((MyType*)this_)->execution_provider_; };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->input_types_.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->input_types_[indice];
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* this_) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->output_types_.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* this_, size_t indice) {
      auto self = reinterpret_cast<const MyType*>(this_);
      return self->output_types_[indice];
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      auto self = reinterpret_cast<MyType*>(op_kernel);
      if (self->exit_fn_) {
        self->exit_fn_(self->custom_handle_);
      }
    };

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp*, size_t) { return INPUT_OUTPUT_REQUIRED; };
  }

  /////////////////////////////  create input tuple ///////////////////////////////

  std::tuple<Args...> CreateInputTupleInvoker(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    return CreateInputTuple<0, 0, Args...>(ort_api, context);
  }

  template <size_t ith_input, size_t ith_output, typename... Ts>
  typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, CustomType*>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    std::tuple<T> current = std::tuple<T>{custom_handle_};
    auto next = CreateInputTuple<ith_input, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, OrtKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    std::tuple<T> current = std::tuple<OrtKernelContext*>{context};
    auto next = CreateInputTuple<ith_input, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  // tensor inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<int64_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int64_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<uint8_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<uint8_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::TensorT<double>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<double>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  // span inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<int64_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int64_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int64_t>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<uint8_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<uint8_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<uint8_t>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, const Custom2::Span<double>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<double>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<double>*>(tensors_.back().get())->AsSpan()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  // scalar inputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, float>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<float>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, int32_t>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int32_t>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, int64_t>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int64_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<int64_t>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, uint8_t>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<uint8_t>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<uint8_t>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, double>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<double>>(ort_api, context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom2::TensorT<double>*>(tensors_.back().get())->AsScalar()};
    auto next = CreateInputTuple<ith_input + 1, ith_output, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  // tensor outputs
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<float>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<float>>(ort_api, context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<int32_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int32_t>>(ort_api, context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<int64_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<int64_t>>(ort_api, context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<uint8_t>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<uint8_t>>(ort_api, context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  typename std::enable_if<std::is_same<T, Custom2::TensorT<double>&>::value, std::tuple<T, Ts...>>::type
  CreateInputTuple(const OrtW::CustomOpApi& ort_api, OrtKernelContext* context) {
    tensors_.push_back(std::make_unique<Custom2::TensorT<double>>(ort_api, context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors_.back())};
    auto next = CreateInputTuple<ith_input, ith_output + 1, Ts...>(ort_api, context);
    return std::tuple_cat(current, next);
  }

  /////////////////////////////  parse args ///////////////////////////////

  template <typename... Ts>
  typename std::enable_if<0 == sizeof...(Ts)>::type
  ParseArgs() {
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, CustomType*>::value>::type
  ParseArgs() {
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, OrtKernelContext*>::value>::type
  ParseArgs() {
    ParseArgs<Ts...>();
  }

  // tensor inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<float>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<int64_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<uint8_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::TensorT<double>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    ParseArgs<Ts...>();
  }

  // span inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<float>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<int32_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<int64_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<uint8_t>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Custom2::Span<double>&>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    ParseArgs<Ts...>();
  }

  // scalar inputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, float>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, int32_t>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, int64_t>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, uint8_t>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, double>::value>::type
  ParseArgs() {
    input_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    ParseArgs<Ts...>();
  }

  // outputs
  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<float>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<int32_t>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<int64_t>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<uint8_t>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);
    ParseArgs<Ts...>();
  }

  template <typename T, typename... Ts>
  typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom2::TensorT<double>&>::value>::type
  ParseArgs() {
    output_types_.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE);
    ParseArgs<Ts...>();
  }

  ////////////////////////////// members //////////////////////////////

  const std::string op_name_;
  const char* execution_provider_;

  const InitFn init_fn_;
  const ComputeFn compute_fn_;
  const ExitFn exit_fn_;

  CustomType* custom_handle_ = {};

  std::vector<TensorPtr> tensors_;
  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;

  const OrtApi* ort_api_{nullptr};
};  // class OrtCustomOpLite

template <typename... Args>
OrtCustomOp* CreateCustomOpT2(const char* op_name,
                              const char* execution_provider,
                              void (*custom_compute_fn)(Args...)) {
  using OrtCustomOpTPtr = OrtCustomOpT2<void, Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider, nullptr, custom_compute_fn, nullptr).release();
}

template <typename... Args>
OrtCustomOp* CreateCustomOpT2(const char* op_name,
                              void (*custom_compute_fn)(Args...)) {
  using OrtCustomOpTPtr = OrtCustomOpT2<void, Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, nullptr, nullptr, custom_compute_fn, nullptr).release();
}

template <typename T, typename... Args>
OrtCustomOp* CreateCustomOpT2(const char* op_name,
                              const char* execution_provider,
                              T* (*custom_init_fn)(const OrtKernelInfo*),
                              void (*custom_compute_fn)(Args...),
                              void (*custom_exit_fn)(T*)) {
  using OrtCustomOpTPtr = OrtCustomOpT2<T, Args...>;
  return std::make_unique<OrtCustomOpTPtr>(op_name, execution_provider, custom_init_fn, custom_compute_fn, custom_exit_fn).release();
}

}  // namespace Custom2
}  // namespace Ort
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_customop.hpp"
#include <optional>
#include <numeric>
// uplevel the version when supported ort version migrates to newer ones
#define SUPPORT_ORT_API_VERSION_TO 13

namespace Ort {
namespace Custom {

class TensorBase {
 public:
  TensorBase(const OrtW::CustomOpApi& api,
             OrtKernelContext& ctx,
             size_t indice,
             bool is_input) : api_(api),
                              ctx_(ctx),
                              indice_(indice),
                              is_input_(is_input) {}

  virtual ~TensorBase() = default;
  operator bool() const {
    return shape_.has_value();
  }
  const std::vector<int64_t>& Shape() const {
    if (shape_.has_value()) {
      return *shape_;
    } else {
      ORTX_CXX_API_THROW("tensor shape is not yet initialized", ORT_RUNTIME_EXCEPTION);
    }
  }
  int64_t NumberOfElement() const {
    if (shape_.has_value()) {
      return std::accumulate(shape_->begin(), shape_->end(), 1LL, std::multiplies<int64_t>());
    } else {
      ORTX_CXX_API_THROW("tensor shape is not yet initialized", ORT_RUNTIME_EXCEPTION);
    }
  }
  std::string Shape2Str() const {
    if (shape_.has_value()) {
      std::string shape_str;
      for (const auto& dim: *shape_) {
        shape_str.append(std::to_string(dim));
        shape_str.append(", ");
      }
      return shape_str;
    } else {
      return "empty";
    }
  }
 protected:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  bool is_input_;
  std::optional<std::vector<int64_t>> shape_;
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
  const T* data() const { return data_; }
};

template <typename T>
class Tensor : public TensorBase {
 public:
  using TT = typename std::remove_reference<T>::type;
  Tensor(const OrtW::CustomOpApi& api,
         OrtKernelContext& ctx,
         size_t indice,
         bool is_input) : TensorBase(api,
                                     ctx,
                                     indice,
                                     is_input) {
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      const_value_ = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value_);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);
    }
  }
  const TT* Data() const {
    return api_.GetTensorData<TT>(const_value_);
  }
  TT* Allocate(const std::vector<int64_t>& shape) {
    if (!data_) {
      OrtValue* out = api_.KernelContext_GetOutput(&ctx_, indice_, shape.data(), shape.size());
      shape_ = shape;
      data_ = api_.GetTensorMutableData<TT>(out);
    }
    return data_;
  }
  const Span<T>& AsSpan() {
    if (!shape_.has_value() || shape_->size() != 1) {
      ORTX_CXX_API_THROW("to get a span, shape must be 1-D, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    span_.Assign(Data(), (*shape_)[0]);
    return span_;
  }
  const T& AsScalar() {
    if (!shape_.has_value() || (shape_->size() == 1 && (*shape_)[0] != 1) || shape_->size() > 1) {
      ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    return *Data();
  }

 private:
  const OrtValue* const_value_{};  // for input
  TT* data_{};                     // for output
  Span<T> span_;
};

template <>
class Tensor<std::string> : public TensorBase {
 public:
  using strings = std::vector<std::string>;

  Tensor(const OrtW::CustomOpApi& api,
         OrtKernelContext& ctx,
         size_t indice,
         bool is_input) : TensorBase(api,
                                     ctx,
                                     indice,
                                     is_input) {
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }

      auto* const_value = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);

      size_t num_chars;
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorDataLength(const_value, &num_chars));
      std::vector<char> chars(num_chars + 1, '\0');
      auto num_strings = NumberOfElement();
      std::vector<size_t> offsets(NumberOfElement());
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorContent(const_value,
                                                                                   (void*)chars.data(),
                                                                                   num_chars,
                                                                                   offsets.data(),
                                                                                   offsets.size()));
      auto upper_bound = static_cast<int64_t>(num_strings) - 1;
      input_strings_.resize(num_strings);
      for (int64_t i = upper_bound; i >= 0; --i) {
        if (i < upper_bound) {
          chars[offsets[i + 1]] = '\0';
        }
        input_strings_[i] = chars.data() + offsets[i];
      }
    }
  }
  const strings& Data() const {
    return input_strings_;
  }
  void SetStringOutput(const strings& ss, const std::vector<int64_t>& dims) {
    std::vector<const char*> raw;
    for (const auto& s : ss) {
      raw.push_back(s.data());
    }
    auto* output = api_.KernelContext_GetOutput(&ctx_, indice_, dims.data(), dims.size());
    OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().FillStringTensor(output, raw.data(), raw.size()));
  }
  void SetStringOutput(const std::vector<const char*>& ss, const std::vector<int64_t>& dims) {
    auto* output = api_.KernelContext_GetOutput(&ctx_, indice_, dims.data(), dims.size());
    OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().FillStringTensor(output, ss.data(), ss.size()));
  }
  const Span<std::string>& AsSpan() {
    ORTX_CXX_API_THROW("span for TensorT of string not implemented", ORT_RUNTIME_EXCEPTION);
  }
  const std::string& AsScalar() {
    if (!shape_.has_value() || (shape_->size() == 1 && (*shape_)[0] != 1) || shape_->size() > 1) {
      ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    return input_strings_[0];
  }

 private:
  std::vector<std::string> input_strings_;  // for input
};

template <>
class Tensor<std::string_view> : public TensorBase {
 public:
  using strings = std::vector<std::string>;
  using string_views = std::vector<std::string_view>;

  Tensor(const OrtW::CustomOpApi& api,
         OrtKernelContext& ctx,
         size_t indice,
         bool is_input) : TensorBase(api,
                                     ctx,
                                     indice,
                                     is_input) {
    if (is_input_) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      auto* const_value = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value);
      shape_ = api_.GetTensorShape(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);

      size_t num_chars;
      OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorDataLength(const_value, &num_chars));
      chars_.resize(num_chars + 1, '\0');

      auto num_strings = static_cast<size_t>(NumberOfElement());
      if (num_strings) {
        std::vector<size_t> offsets(num_strings);
        OrtW::ThrowOnError(api_.GetOrtApi(), api_.GetOrtApi().GetStringTensorContent(const_value,
                                                                                     (void*)chars_.data(),
                                                                                     num_chars,
                                                                                     offsets.data(),
                                                                                     offsets.size()));
        offsets.push_back(num_chars);
        for (size_t i = 0; i < num_strings; ++i) {
          input_string_views_.emplace_back(chars_.data() + offsets[i], offsets[i + 1] - offsets[i]);
        }
      }
    }
  }
  int64_t NumberOfElement() const {
    if (shape_.has_value()) {
      return std::accumulate(shape_->begin(), shape_->end(), 1ULL, std::multiplies<int64_t>());
    } else {
      return 0;
    }
  }
  const string_views& Data() const {
    return input_string_views_;
  }
  const Span<std::string_view>& AsSpan() {
    ORTX_CXX_API_THROW("span for TensorT of string view not implemented", ORT_RUNTIME_EXCEPTION);
  }
  std::string_view AsScalar() {
    if (!shape_.has_value() || (shape_->size() == 1 && (*shape_)[0] != 1) || shape_->size() > 1) {
      ORTX_CXX_API_THROW("to get a scalar, shape must be {1}, actual shape: " + Shape2Str(), ORT_RUNTIME_EXCEPTION);
    }
    return input_string_views_[0];
  }

 private:
  std::vector<char> chars_;                           // for input
  std::vector<std::string_view> input_string_views_;  // for input
};

using TensorPtr = std::unique_ptr<Custom::TensorBase>;

struct OrtLiteCustomOp : public OrtCustomOp {
  using ConstOptionalFloatTensor = std::optional<const Custom::Tensor<float>&>;
  using OptionalFloatTensor = std::optional<Custom::Tensor<float>>;

  // CreateTuple
  template <size_t ith_input, size_t ith_output, typename... Ts>
  static typename std::enable_if<sizeof...(Ts) == 0, std::tuple<>>::type
  CreateTuple(const OrtW::CustomOpApi*, OrtKernelContext*, std::vector<TensorPtr>&, size_t, size_t, const std::string&) {
    return std::make_tuple();
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, OrtKernelContext*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    std::tuple<T> current = std::tuple<OrtKernelContext*>{context};
    auto next = CreateTuple<ith_input, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

#define CREATE_TUPLE_INPUT(data_type)                                                                                                                                 \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, const Custom::Tensor<data_type>*>::value, std::tuple<T, Ts...>>::type                                                \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};                                                                                 \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, const Custom::Tensor<data_type>&>::value, std::tuple<T, Ts...>>::type                                                \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};                                                                                \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<const Custom::Tensor<data_type>*>>::value, std::tuple<T, Ts...>>::type                                 \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_input < num_input) {                                                                                                                                      \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                \
      std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())};                                                      \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    } else {                                                                                                                                                          \
      std::tuple<T> current = std::tuple<T>{};                                                                                                                        \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    }                                                                                                                                                                 \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, const Custom::Span<data_type>*>::value, std::tuple<T, Ts...>>::type                                                  \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if ("CPUExecutionProvider" != ep) {                                                                                                                               \
      ORTX_CXX_API_THROW("span input could only be applied to CPU EP", ORT_FAIL);                                                                                     \
    }                                                                                                                                                                 \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    std::tuple<T> current = std::tuple<T>{&reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsSpan()};                                             \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, const Custom::Span<data_type>&>::value, std::tuple<T, Ts...>>::type                                                  \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if ("CPUExecutionProvider" != ep) {                                                                                                                               \
      ORTX_CXX_API_THROW("span input could only be applied to CPU EP", ORT_FAIL);                                                                                     \
    }                                                                                                                                                                 \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsSpan()};                                              \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<const Custom::Span<data_type>*>>::value, std::tuple<T, Ts...>>::type                                   \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_input < num_input) {                                                                                                                                      \
      if ("CPUExecutionProvider" != ep) {                                                                                                                             \
        ORTX_CXX_API_THROW("span input could only be applied to CPU EP", ORT_FAIL);                                                                                   \
      }                                                                                                                                                               \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                \
      std::tuple<T> current = std::tuple<T>{&reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsSpan()};                                           \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    } else {                                                                                                                                                          \
      std::tuple<T> current = std::tuple<T>{};                                                                                                                        \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    }                                                                                                                                                                 \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, data_type>::value, std::tuple<T, Ts...>>::type                                                                       \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if ("CPUExecutionProvider" != ep) {                                                                                                                               \
      ORTX_CXX_API_THROW("scalar input could only be applied to CPU EP", ORT_FAIL);                                                                                   \
    }                                                                                                                                                                 \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsScalar()};                                            \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<data_type>>::value, std::tuple<T, Ts...>>::type                                                        \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_input < num_input) {                                                                                                                                      \
      if ("CPUExecutionProvider" != ep) {                                                                                                                             \
        ORTX_CXX_API_THROW("scalar input could only be applied to CPU EP", ORT_FAIL);                                                                                 \
      }                                                                                                                                                               \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                \
      std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsScalar()};                                          \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    } else {                                                                                                                                                          \
      std::tuple<T> current = std::tuple<T>{};                                                                                                                        \
      auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    }                                                                                                                                                                 \
  }
#define CREATE_TUPLE_OUTPUT(data_type)                                                                                                                                \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, Custom::Tensor<data_type>*>::value, std::tuple<T, Ts...>>::type                                                      \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_output, false));                                                                \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};                                                                                 \
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, Custom::Tensor<data_type>&>::value, std::tuple<T, Ts...>>::type                                                      \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_output, false));                                                                \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};                                                                                \
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<Custom::Tensor<data_type>*>>::value, std::tuple<T, Ts...>>::type                                       \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_output < num_output) {                                                                                                                                    \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_output, false));                                                              \
      std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())};                                                      \
      auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    } else {                                                                                                                                                          \
      std::tuple<T> current = std::tuple<T>{};                                                                                                                        \
      auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);                                                    \
      return std::tuple_cat(current, next);                                                                                                                           \
    }                                                                                                                                                                 \
  }
#define CREATE_TUPLE(data_type) \
  CREATE_TUPLE_INPUT(data_type) \
  CREATE_TUPLE_OUTPUT(data_type)

  CREATE_TUPLE(bool)
  CREATE_TUPLE(float)
  CREATE_TUPLE(double)
  CREATE_TUPLE(int8_t)
  CREATE_TUPLE(int16_t)
  CREATE_TUPLE(int32_t)
  CREATE_TUPLE(int64_t)
  CREATE_TUPLE(uint8_t)
  CREATE_TUPLE(uint16_t)
  CREATE_TUPLE(uint32_t)
  CREATE_TUPLE(uint64_t)
  CREATE_TUPLE(std::string)
  CREATE_TUPLE_INPUT(std::string_view)

  // ParseArgs ...
  template <typename... Ts>
  static typename std::enable_if<0 == sizeof...(Ts)>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>&, std::vector<ONNXTensorElementDataType>&) {
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, OrtKernelContext*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }

#define PARSE_INPUT_BASE(pack_type, onnx_type)                                                                           \
  template <typename T, typename... Ts>                                                                                  \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, pack_type>::value>::type                          \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) { \
    input_types.push_back(onnx_type);                                                                                    \
    ParseArgs<Ts...>(input_types, output_types);                                                                         \
  }                                                                                                                      \
  template <typename T, typename... Ts>                                                                                  \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, std::optional<pack_type>>::value>::type           \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) { \
    input_types.push_back(onnx_type);                                                                                    \
    ParseArgs<Ts...>(input_types, output_types);                                                                         \
  }

#define PARSE_INPUT(data_type, onnx_type)                       \
  PARSE_INPUT_BASE(const Custom::Tensor<data_type>*, onnx_type) \
  PARSE_INPUT_BASE(const Custom::Tensor<data_type>&, onnx_type) \
  PARSE_INPUT_BASE(const Custom::Span<data_type>*, onnx_type)   \
  PARSE_INPUT_BASE(const Custom::Span<data_type>&, onnx_type)   \
  PARSE_INPUT_BASE(data_type, onnx_type)

#define PARSE_OUTPUT(data_type, onnx_type)                                                                                      \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom::Tensor<data_type>*>::value>::type                \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }                                                                                                                             \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Custom::Tensor<data_type>&>::value>::type                \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }                                                                                                                             \
  template <typename T, typename... Ts>                                                                                         \
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, std::optional<Custom::Tensor<data_type>*>>::value>::type \
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {        \
    output_types.push_back(onnx_type);                                                                                          \
    ParseArgs<Ts...>(input_types, output_types);                                                                                \
  }

#define PARSE_ARGS(data_type, onnx_type) \
  PARSE_INPUT(data_type, onnx_type)      \
  PARSE_OUTPUT(data_type, onnx_type)

  PARSE_ARGS(bool, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL)
  PARSE_ARGS(float, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  PARSE_ARGS(double, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE)
  PARSE_ARGS(int8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
  PARSE_ARGS(int16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16)
  PARSE_ARGS(int32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
  PARSE_ARGS(int64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64)
  PARSE_ARGS(uint8_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8)
  PARSE_ARGS(uint16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16)
  PARSE_ARGS(uint32_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32)
  PARSE_ARGS(uint64_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64)
  PARSE_ARGS(std::string, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)
  PARSE_ARGS(std::string_view, ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING)  // todo - remove string_view output

  OrtLiteCustomOp(const char* op_name,
                  const char* execution_provider) : op_name_(op_name),
                                                    execution_provider_(execution_provider) {
    OrtCustomOp::version = MIN_ORT_VERSION_SUPPORTED;

    OrtCustomOp::GetName = [](const OrtCustomOp* op) { return static_cast<const OrtLiteCustomOp*>(op)->op_name_.c_str(); };
    OrtCustomOp::GetExecutionProviderType = [](const OrtCustomOp* op) { return ((OrtLiteCustomOp*)op)->execution_provider_.c_str(); };

    OrtCustomOp::GetInputTypeCount = [](const OrtCustomOp* op) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->input_types_.size();
    };

    OrtCustomOp::GetInputType = [](const OrtCustomOp* op, size_t indice) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->input_types_[indice];
    };

    OrtCustomOp::GetOutputTypeCount = [](const OrtCustomOp* op) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->output_types_.size();
    };

    OrtCustomOp::GetOutputType = [](const OrtCustomOp* op, size_t indice) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return self->output_types_[indice];
    };

    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp*, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };
  }

  const std::string op_name_;
  const std::string execution_provider_;

  std::vector<ONNXTensorElementDataType> input_types_;
  std::vector<ONNXTensorElementDataType> output_types_;
};

template <typename... Args>
struct OrtLiteCustomFunc : public OrtLiteCustomOp {
  using ComputeFn = void (*)(Args...);
  using MyType = OrtLiteCustomFunc<Args...>;

  struct Kernel {
    ComputeFn compute_fn_{};
    std::string ep_{};
    std::unique_ptr<OrtW::CustomOpApi> api_;
  };

  OrtLiteCustomFunc(const char* op_name,
                    const char* execution_provider,
                    ComputeFn compute_fn) : OrtLiteCustomOp(op_name, execution_provider),
                                            compute_fn_(compute_fn) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto kernel = reinterpret_cast<Kernel*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->api_.get(),
                                          context,
                                          tensors,
                                          kernel->api_->KernelContext_GetInputCount(context),
                                          kernel->api_->KernelContext_GetOutputCount(context),
                                          kernel->ep_);
      std::apply([kernel](Args const&... t_args) { kernel->compute_fn_(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort_api, const OrtKernelInfo* info) {
      auto kernel = std::make_unique<Kernel>();
      auto self = static_cast<const OrtLiteCustomFunc*>(this_);
      kernel->compute_fn_ = self->compute_fn_;
      kernel->ep_ = self->execution_provider_;
      kernel->api_ = std::make_unique<OrtW::CustomOpApi>(*ort_api);
      return reinterpret_cast<void*>(kernel.release());
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      delete reinterpret_cast<Kernel*>(op_kernel);
    };
  }

  ComputeFn compute_fn_;
};

template <typename CustomOp>
struct OrtLiteCustomStruct : public OrtLiteCustomOp {
  template <typename... Args>
  using CustomComputeFn = void (CustomOp::*)(Args...);
  using MyType = OrtLiteCustomStruct<CustomOp>;

  struct Kernel {
    std::unique_ptr<CustomOp> custom_op_;
    std::string ep_{};
    std::unique_ptr<OrtW::CustomOpApi> api_;
  };

  OrtLiteCustomStruct(const char* op_name,
                      const char* execution_provider) : OrtLiteCustomOp(op_name,
                                                                        execution_provider) {
    init(&CustomOp::Compute);
  }

  template <typename... Args>
  void init(CustomComputeFn<Args...>) {
    ParseArgs<Args...>(input_types_, output_types_);

    OrtCustomOp::KernelCompute = [](void* op_kernel, OrtKernelContext* context) {
      auto kernel = reinterpret_cast<Kernel*>(op_kernel);
      std::vector<TensorPtr> tensors;
      auto t = CreateTuple<0, 0, Args...>(kernel->api_.get(),
                                          context,
                                          tensors,
                                          kernel->api_->KernelContext_GetInputCount(context),
                                          kernel->api_->KernelContext_GetOutputCount(context),
                                          kernel->ep_);
      std::apply([kernel](Args const&... t_args) { kernel->custom_op_->Compute(t_args...); }, t);
    };

    OrtCustomOp::CreateKernel = [](const OrtCustomOp* this_, const OrtApi* ort_api, const OrtKernelInfo* info) {
      auto kernel = std::make_unique<Kernel>();
      kernel->custom_op_ = std::make_unique<CustomOp>(*ort_api, *info);
      auto self = static_cast<const MyType*>(this_);
      kernel->ep_ = self->execution_provider_;
      kernel->api_ = std::make_unique<OrtW::CustomOpApi>(*ort_api);
      return reinterpret_cast<void*>(kernel.release());
    };

    OrtCustomOp::KernelDestroy = [](void* op_kernel) {
      delete reinterpret_cast<Kernel*>(op_kernel);
    };
  }
};

template <typename... Args>
OrtLiteCustomOp* CreateLiteCustomOp(const char* op_name,
                                    const char* execution_provider,
                                    void (*custom_compute_fn)(Args...)) {
  using LiteOp = OrtLiteCustomFunc<Args...>;
  return std::make_unique<LiteOp>(op_name, execution_provider, custom_compute_fn).release();
}

template <typename CustomOp>
OrtLiteCustomOp* CreateLiteCustomOp(const char* op_name,
                                    const char* execution_provider) {
  using LiteOp = OrtLiteCustomStruct<CustomOp>;
  return std::make_unique<LiteOp>(op_name, execution_provider).release();
}

}  // namespace Custom
}  // namespace Ort
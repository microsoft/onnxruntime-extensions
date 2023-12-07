// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "onnxruntime_customop.hpp"
#include "onnxruntime_f16.h"
#include <optional>
#include <numeric>
//#ifdef USE_CUDA
//#include <cuda_bf16.h>
//#endif

namespace Ort {
namespace Custom {

//// MFloat16
//struct Float16_t : onnxruntime_float16::Float16Impl<Float16_t> {
// private:
//  constexpr explicit Float16_t(uint16_t v) noexcept { val = v; }
//
// public:
//  using Base = onnxruntime_float16::Float16Impl<Float16_t>;
//
//  Float16_t() = default;
//
//  constexpr static Float16_t FromBits(uint16_t v) noexcept { return Float16_t(v); }
//
//  explicit Float16_t(float v) noexcept { val = Base::ToUint16Impl(v); }
//
//  float ToFloat() const noexcept { return Base::ToFloatImpl(); }
//
//  using Base::Abs;
//  using Base::AreZero;
//  using Base::IsFinite;
//  using Base::IsInfinity;
//  using Base::IsNaN;
//  using Base::IsNaNOrZero;
//  using Base::IsNegative;
//  using Base::IsNegativeInfinity;
//  using Base::IsNormal;
//  using Base::IsPositiveInfinity;
//  using Base::IsSubnormal;
//  using Base::Negate;
//
//  explicit operator float() const noexcept { return ToFloat(); }
//
//  using Base::operator==;
//  using Base::operator!=;
//  using Base::operator<;
//};

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
  ONNXTensorElementDataType Type() const {
    return type_;
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
      for (const auto& dim : *shape_) {
        shape_str.append(std::to_string(dim));
        shape_str.append(", ");
      }
      return shape_str;
    } else {
      return "empty";
    }
  }
  bool IsCpuTensor() const {
    return strcmp("Cpu", mem_type_) == 0;
  }
  virtual const void* DataRaw() const = 0;
  virtual size_t SizeInBytes() const = 0;

 protected:
  const OrtW::CustomOpApi& api_;
  OrtKernelContext& ctx_;
  size_t indice_;
  bool is_input_;
  std::optional<std::vector<int64_t>> shape_;
  ONNXTensorElementDataType type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  const char* mem_type_ = "Cpu";
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORTC_HOST_DEVICE __host__ __device__
#else
#define ORTC_HOST_DEVICE
#endif

//// BFloat16
//struct BFloat16 : onnxruntime_float16::BFloat16Impl<BFloat16> {
//  using Base = onnxruntime_float16::BFloat16Impl<BFloat16>;
//
//#if defined(__HIP__)
//  ORTC_HOST_DEVICE BFloat16() = default;
//#else
//  BFloat16() = default;
//#endif
//
//  struct FromBitsT {};
//  static constexpr ORTC_HOST_DEVICE FromBitsT FromBits() noexcept { return FromBitsT(); }
//  constexpr ORTC_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) noexcept { val = bits; }
//
//  static constexpr ORTC_HOST_DEVICE BFloat16 FromBits(uint16_t bits) noexcept {
//    return BFloat16(bits, FromBits());
//  }
//
//  inline ORTC_HOST_DEVICE BFloat16(float v) noexcept {
//#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//    val = __bfloat16_as_ushort(__float2bfloat16(v));
//#elif defined(__HIP__)
//    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
//    if (v != v) {  // isnan
//      val = UINT16_C(0x7FC0);
//    } else {
//      union {
//        uint32_t U32;
//        float F32;
//      };
//
//      F32 = v;
//      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
//      val = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
//    }
//#else
//
//    // Use C isnan to work both in host and device
//    if (::isnan(v)) {
//      val = kPositiveQNaNBits;
//    } else {
//      auto get_msb_half = [](float fl) {
//        uint16_t result;
//        if constexpr (onnxruntime_float16::detail::endian::native == onnxruntime_float16::detail::endian::little) {
//          std::memcpy(&result, reinterpret_cast<char*>(&fl) + sizeof(uint16_t), sizeof(uint16_t));
//        } else {
//          std::memcpy(&result, &fl, sizeof(uint16_t));
//        }
//        return result;
//      };
//
//      uint16_t upper_bits = get_msb_half(v);
//      union {
//        uint32_t U32;
//        float F32;
//      };
//      F32 = v;
//      U32 += (upper_bits & 1) + kRoundToNearest;
//      val = get_msb_half(F32);
//    }
//#endif
//  }
//
//  inline ORTC_HOST_DEVICE float ToFloat() const noexcept {
//#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
//    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
//#elif defined(__HIP__)
//    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
//    float result = 0;
//    uint32_t tmp = val;
//    tmp <<= 16;
//    float* tempRes = reinterpret_cast<float*>(&tmp);
//    result = *tempRes;
//    return result;
//#else
//
//    if (IsNaNHostDevice()) {
//      return std::numeric_limits<float>::quiet_NaN();
//    }
//
//    float result = 0;
//    char* const first = reinterpret_cast<char*>(&result);
//    if constexpr (onnxruntime_float16::detail::endian::native == onnxruntime_float16::detail::endian::little) {
//      char* const second = first + sizeof(uint16_t);
//      std::memcpy(second, &val, sizeof(uint16_t));
//    } else {
//      std::memcpy(first, &val, sizeof(uint16_t));
//    }
//    return result;
//#endif
//  }
//
//  static const BFloat16 NaN;
//  static const BFloat16 NegativeNaN;
//  static const BFloat16 Infinity;
//  static const BFloat16 NegativeInfinity;
//  static const BFloat16 Epsilon;
//  static const BFloat16 MinValue;
//  static const BFloat16 MaxValue;
//  static const BFloat16 Zero;
//  static const BFloat16 One;
//  static const BFloat16 MinusOne;
//
//  using Base::IsNegative;
//
//  using Base::IsNaN;
//
//  using Base::IsFinite;
//
//  using Base::IsPositiveInfinity;
//
//  using Base::IsNegativeInfinity;
//
//  using Base::IsInfinity;
//
//  using Base::IsNaNOrZero;
//
//  using Base::IsNormal;
//
//  using Base::IsSubnormal;
//
//  using Base::Abs;
//
//  using Base::Negate;
//
//  ORTC_HOST_DEVICE operator float() const noexcept { return ToFloat(); }
//
//#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
//  ORTC_HOST_DEVICE BFloat16(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); }
//  explicit ORTC_HOST_DEVICE operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&val); }
//#endif
//
//  ORTC_HOST_DEVICE bool operator==(const BFloat16& rhs) const noexcept {
//    if (IsNaNHostDevice() || rhs.IsNaNHostDevice()) {
//      // IEEE defines that NaN is not equal to anything, including itself.
//      return false;
//    }
//    return val == rhs.val;
//  }
//
//  ORTC_HOST_DEVICE bool operator!=(const BFloat16& rhs) const noexcept {
//    return !(*this == rhs);
//  }
//
//  ORTC_HOST_DEVICE bool operator<(const BFloat16& rhs) const noexcept {
//    if (IsNaNHostDevice() || rhs.IsNaNHostDevice()) {
//      // IEEE defines that NaN is unordered with respect to everything, including itself.
//      return false;
//    }
//
//    const bool left_is_negative = IsNegativeHostDevice();
//    if (left_is_negative != rhs.IsNegativeHostDevice()) {
//      // When the signs of left and right differ, we know that left is less than right if it is
//      // the negative value. The exception to this is if both values are zero, in which case IEEE
//      // says they should be equal, even if the signs differ.
//      return left_is_negative && !AreZeroHostDevice(*this, rhs);
//    }
//    return (val != rhs.val) && ((val < rhs.val) ^ left_is_negative);
//  }
//
//  ORTC_HOST_DEVICE bool IsNegativeHostDevice() const noexcept {
//    return (val & kSignMask) != 0;
//  }
//
//  ORTC_HOST_DEVICE bool IsNaNHostDevice() const noexcept {
//    return static_cast<uint16_t>(val & ~kSignMask) > kPositiveInfinityBits;
//  }
//
//  ORTC_HOST_DEVICE static bool AreZeroHostDevice(const BFloat16Impl& lhs, const BFloat16Impl& rhs) noexcept {
//    // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
//    // for two values by or'ing the private bits together and stripping the sign. They are both zero,
//    // and therefore equivalent, if the resulting value is still zero.
//    return static_cast<uint16_t>((lhs.val | rhs.val) & ~kSignMask) == 0;
//  }
//};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

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

template <>
struct Span<Float16_t> {
  const Float16_t* data_ = {};
  size_t size_ = {};
  void Assign(const Float16_t* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  Float16_t operator[](size_t indice) const {
    return data_[indice];
  }
  const Float16_t* data() const { return data_; }
};

template <>
struct Span<BFloat16> {
  const BFloat16* data_ = {};
  size_t size_ = {};
  void Assign(const BFloat16* data, size_t size) {
    data_ = data;
    size_ = size;
  }
  size_t size() const { return size_; }
  BFloat16 operator[](size_t indice) const {
    return data_[indice];
  }
  const BFloat16* data() const { return data_; }
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
      type_ = api_.GetTensorElementType(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);
      const OrtMemoryInfo* mem_info = {};
      api_.ThrowOnError(api_.GetOrtApi().GetTensorMemoryInfo(const_value_, &mem_info));
      if (mem_info) {
        api_.ThrowOnError(api.GetOrtApi().MemoryInfoGetName(mem_info, &mem_type_));
      }
    }
  }
  const TT* Data() const {
    return api_.GetTensorData<TT>(const_value_);
  }

  const void* DataRaw() const override {
    return reinterpret_cast<const void*>(Data());
  }

  size_t SizeInBytes() const override {
    return NumberOfElement() * sizeof(TT);
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
      type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
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
  const void* DataRaw() const override {
    if (input_strings_.size() != 1) {
      ORTX_CXX_API_THROW("DataRaw() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return reinterpret_cast<const void*>(input_strings_[0].c_str());
  }
  size_t SizeInBytes() const override {
    if (input_strings_.size() != 1) {
      ORTX_CXX_API_THROW("SizeInBytes() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return input_strings_[0].size();
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
      type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
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
  const void* DataRaw() const override {
    if (input_string_views_.size() != 1) {
      ORTX_CXX_API_THROW("DataRaw() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return reinterpret_cast<const void*>(input_string_views_[0].data());
  }
  size_t SizeInBytes() const override {
    if (input_string_views_.size() != 1) {
      ORTX_CXX_API_THROW("SizeInBytes() only applies to string scalar", ORT_RUNTIME_EXCEPTION);
    }
    return input_string_views_[0].size();
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

template <>
struct Tensor<Float16_t> : public TensorBase {
  Tensor(const OrtW::CustomOpApi& api,
         OrtKernelContext& ctx,
         size_t indice,
         bool is_input) : TensorBase(api,
                                     ctx,
                                     indice,
                                     is_input) {
    type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (is_input_) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      const_value_ = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value_);
      shape_ = api_.GetTensorShape(info);
      type_ = api_.GetTensorElementType(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);
      const OrtMemoryInfo* mem_info = {};
      api_.ThrowOnError(api_.GetOrtApi().GetTensorMemoryInfo(const_value_, &mem_info));
      if (mem_info) {
        api_.ThrowOnError(api.GetOrtApi().MemoryInfoGetName(mem_info, &mem_type_));
      }
    }
  }

  const Float16_t* Data() const {
    return reinterpret_cast<const Float16_t*>(api_.GetTensorData<uint16_t>(const_value_));
  }

  Float16_t* Allocate(const std::vector<int64_t>& shape) {
    if (!data_) {
      OrtValue* out = api_.KernelContext_GetOutput(&ctx_, indice_, shape.data(), shape.size());
      shape_ = shape;
      data_ = reinterpret_cast<Float16_t*>(api_.GetTensorMutableData<uint16_t>(out));
    }
    return data_;
  }

  const Span<Float16_t>& AsSpan() {
    ORTX_CXX_API_THROW("AsSpan for Float16_t not implemented", ORT_RUNTIME_EXCEPTION);
  }

  const Float16_t& AsScalar() {
    ORTX_CXX_API_THROW("AsScalar for Float16_t not implemented", ORT_RUNTIME_EXCEPTION);
  }

  const void* DataRaw() const override {
    return reinterpret_cast<const void*>(Data());
  }

  virtual size_t SizeInBytes() const override {
    return NumberOfElement() * sizeof(uint16_t);
  }

 private:
  const OrtValue* const_value_{};  // for input
  Float16_t* data_{};              // for output
};

template <>
struct Tensor<BFloat16> : public TensorBase {
  Tensor(const OrtW::CustomOpApi& api,
         OrtKernelContext& ctx,
         size_t indice,
         bool is_input) : TensorBase(api,
                                     ctx,
                                     indice,
                                     is_input) {
    type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    if (is_input_) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      if (indice >= input_count) {
        ORTX_CXX_API_THROW("invalid indice", ORT_RUNTIME_EXCEPTION);
      }
      const_value_ = api_.KernelContext_GetInput(&ctx_, indice);
      auto* info = api_.GetTensorTypeAndShape(const_value_);
      shape_ = api_.GetTensorShape(info);
      type_ = api_.GetTensorElementType(info);
      api_.ReleaseTensorTypeAndShapeInfo(info);
      const OrtMemoryInfo* mem_info = {};
      api_.ThrowOnError(api_.GetOrtApi().GetTensorMemoryInfo(const_value_, &mem_info));
      if (mem_info) {
        api_.ThrowOnError(api.GetOrtApi().MemoryInfoGetName(mem_info, &mem_type_));
      }
    }
  }

  const BFloat16* Data() const {
    return reinterpret_cast<const BFloat16*>(api_.GetTensorData<uint16_t>(const_value_));
  }

  BFloat16* Allocate(const std::vector<int64_t>& shape) {
    if (!data_) {
      OrtValue* out = api_.KernelContext_GetOutput(&ctx_, indice_, shape.data(), shape.size());
      shape_ = shape;
      data_ = reinterpret_cast<BFloat16*>(api_.GetTensorMutableData<uint16_t>(out));
    }
    return data_;
  }

  const Span<BFloat16>& AsSpan() {
    ORTX_CXX_API_THROW("AsSpan for BFloat16 not implemented", ORT_RUNTIME_EXCEPTION);
  }

  const BFloat16& AsScalar() {
    ORTX_CXX_API_THROW("AsScalar for BFloat16 not implemented", ORT_RUNTIME_EXCEPTION);
  }

  const void* DataRaw() const override {
    return reinterpret_cast<const void*>(Data());
  }

  virtual size_t SizeInBytes() const override {
    return NumberOfElement() * sizeof(uint16_t);
  }

 private:
  const OrtValue* const_value_{};  // for input
  BFloat16* data_{};               // for output
};

using TensorPtr = std::unique_ptr<Custom::TensorBase>;
using TensorPtrs = std::vector<TensorPtr>;

// Represent variadic input or output
struct Variadic : public TensorBase {
  Variadic(const OrtW::CustomOpApi& api,
           OrtKernelContext& ctx,
           size_t indice,
           bool is_input) : TensorBase(api,
                                       ctx,
                                       indice,
                                       is_input) {
#if ORT_API_VERSION < 14
    ORTX_CXX_API_THROW("Variadic input or output only supported after onnxruntime 1.14", ORT_RUNTIME_EXCEPTION);
#endif
    if (is_input) {
      auto input_count = api_.KernelContext_GetInputCount(&ctx_);
      for (size_t ith_input = 0; ith_input < input_count; ++ith_input) {
        auto* const_value = api_.KernelContext_GetInput(&ctx_, ith_input);
        auto* info = api_.GetTensorTypeAndShape(const_value);
        auto type = api_.GetTensorElementType(info);
        api_.ReleaseTensorTypeAndShapeInfo(info);
        TensorPtr tensor;
        switch (type) {
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            tensor = std::make_unique<Custom::Tensor<bool>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
            tensor = std::make_unique<Custom::Tensor<float>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
            tensor = std::make_unique<Custom::Tensor<double>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
            tensor = std::make_unique<Custom::Tensor<uint8_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
            tensor = std::make_unique<Custom::Tensor<int8_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
            tensor = std::make_unique<Custom::Tensor<uint16_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
            tensor = std::make_unique<Custom::Tensor<int16_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            tensor = std::make_unique<Custom::Tensor<uint32_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            tensor = std::make_unique<Custom::Tensor<int32_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
            tensor = std::make_unique<Custom::Tensor<uint64_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            tensor = std::make_unique<Custom::Tensor<int64_t>>(api, ctx, ith_input, true);
            break;
          case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
            tensor = std::make_unique<Custom::Tensor<std::string>>(api, ctx, ith_input, true);
            break;
          default:
            ORTX_CXX_API_THROW("unknow input type", ORT_RUNTIME_EXCEPTION);
            break;
        }
        tensors_.emplace_back(tensor.release());
      }  // for
    } else {
      // a Variadic used for output is populated by the Compute so leave tensors_ empty here
    }
  }
  template <typename T>
  T* AllocateOutput(size_t ith_output, const std::vector<int64_t>& shape) {
    auto tensor = std::make_unique<Tensor<T>>(api_, ctx_, ith_output, false);
    auto raw_output = tensor.get()->Allocate(shape);
    tensors_.emplace_back(tensor.release());
    return raw_output;
  }
  Tensor<std::string>& AllocateStringTensor(size_t ith_output) {
    auto tensor = std::make_unique<Tensor<std::string>>(api_, ctx_, ith_output, false);
    Tensor<std::string>& output = *tensor;
    tensors_.emplace_back(tensor.release());
    return output;
  }
  const void* DataRaw() const override {
    ORTX_CXX_API_THROW("DataRaw() cannot be applied to Variadic", ORT_RUNTIME_EXCEPTION);
    return nullptr;
  }
  size_t SizeInBytes() const override {
    ORTX_CXX_API_THROW("SizeInBytes() cannot be applied to Variadic", ORT_RUNTIME_EXCEPTION);
    return 0;
  }
  size_t Size() const {
    return tensors_.size();
  }
  const TensorPtr& operator[](size_t indice) const {
    return tensors_.at(indice);
  }

 private:
  TensorPtrs tensors_;
};

#ifdef USE_CUDA

enum CudaResource {
  cuda_handle_t = 10000,
};

struct CudaContext {
  static const int cuda_resource_ver = 1;
  void Init(const OrtW::CustomOpApi& api, const OrtKernelContext& ctx) {
    const auto& ort_api = api.GetOrtApi();
    ort_api.KernelContext_GetResource(&ctx, cuda_resource_ver, CudaResource::cuda_handle_t, &cuda_stream);
    if (!cuda_stream) {
      ORTX_CXX_API_THROW("Failed to fetch cuda stream from context", ORT_RUNTIME_EXCEPTION);
    }
  }
  void* cuda_stream = {};
};

#endif

//using mf16_t = uint16_t;

struct OrtLiteCustomOp : public OrtCustomOp {
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

#ifdef USE_CUDA
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const CudaContext&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    thread_local CudaContext cuda_context;
    cuda_context.Init(*api, *context);
    std::tuple<T> current = std::tuple<const CudaContext&>{cuda_context};
    auto next = CreateTuple<ith_input, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
#endif

#if ORT_API_VERSION >= 14
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Variadic*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, const Variadic&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_input, true));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Variadic*>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }

  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>
  static typename std::enable_if<std::is_same<T, Variadic&>::value, std::tuple<T, Ts...>>::type
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) {
    tensors.push_back(std::make_unique<Variadic>(*api, *context, ith_output, false));
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<T>(*tensors.back().get())};
    auto next = CreateTuple<ith_input, ith_output + 1, Ts...>(api, context, tensors, num_input, num_output, ep);
    return std::tuple_cat(current, next);
  }
#endif

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
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    if (!tensors.back()->IsCpuTensor()) {                                                                                                                             \
      ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);                                                                                 \
    }                                                                                                                                                                 \
    std::tuple<T> current = std::tuple<T>{&reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsSpan()};                                             \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, const Custom::Span<data_type>&>::value, std::tuple<T, Ts...>>::type                                                  \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    if (!tensors.back()->IsCpuTensor()) {                                                                                                                             \
      ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);                                                                                 \
    }                                                                                                                                                                 \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsSpan()};                                              \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<const Custom::Span<data_type>*>>::value, std::tuple<T, Ts...>>::type                                   \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_input < num_input) {                                                                                                                                      \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                \
      if (!tensors.back()->IsCpuTensor()) {                                                                                                                           \
        ORTX_CXX_API_THROW("span input could only be applied to CPU tensor", ORT_FAIL);                                                                               \
      }                                                                                                                                                               \
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
    tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                  \
    if (!tensors.back()->IsCpuTensor()) {                                                                                                                             \
      ORTX_CXX_API_THROW("scalar input could only be applied to CPU tensor", ORT_FAIL);                                                                               \
    }                                                                                                                                                                 \
    std::tuple<T> current = std::tuple<T>{reinterpret_cast<Custom::Tensor<data_type>*>(tensors.back().get())->AsScalar()};                                            \
    auto next = CreateTuple<ith_input + 1, ith_output, Ts...>(api, context, tensors, num_input, num_output, ep);                                                      \
    return std::tuple_cat(current, next);                                                                                                                             \
  }                                                                                                                                                                   \
  template <size_t ith_input, size_t ith_output, typename T, typename... Ts>                                                                                          \
  static typename std::enable_if<std::is_same<T, std::optional<data_type>>::value, std::tuple<T, Ts...>>::type                                                        \
  CreateTuple(const OrtW::CustomOpApi* api, OrtKernelContext* context, std::vector<TensorPtr>& tensors, size_t num_input, size_t num_output, const std::string& ep) { \
    if (ith_input < num_input) {                                                                                                                                      \
      tensors.push_back(std::make_unique<Custom::Tensor<data_type>>(*api, *context, ith_input, true));                                                                \
      if (!tensors.back()->IsCpuTensor()) {                                                                                                                           \
        ORTX_CXX_API_THROW("scalar input could only be applied to CPU tensor", ORT_FAIL);                                                                             \
      }                                                                                                                                                               \
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
  CREATE_TUPLE(Float16_t)
  CREATE_TUPLE(BFloat16)
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

#ifdef USE_CUDA
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const CudaContext&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    ParseArgs<Ts...>(input_types, output_types);
  }
#endif

#if ORT_API_VERSION >= 14
  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Variadic&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!input_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic input, only one input is allowed", ORT_RUNTIME_EXCEPTION);
    }
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, const Variadic*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!input_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic input, only one input is allowed", ORT_RUNTIME_EXCEPTION);
    }
    input_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Variadic&>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!output_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic output, only one output is allowed", ORT_RUNTIME_EXCEPTION);
    }
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }

  template <typename T, typename... Ts>
  static typename std::enable_if<0 <= sizeof...(Ts) && std::is_same<T, Variadic*>::value>::type
  ParseArgs(std::vector<ONNXTensorElementDataType>& input_types, std::vector<ONNXTensorElementDataType>& output_types) {
    if (!output_types.empty()) {
      ORTX_CXX_API_THROW("for op has variadic output, only one output is allowed", ORT_RUNTIME_EXCEPTION);
    }
    output_types.push_back(ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED);
    ParseArgs<Ts...>(input_types, output_types);
  }
#endif

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
  PARSE_ARGS(Float16_t, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
  PARSE_ARGS(BFloat16, ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16)
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
    // Zero out OrtCustomOp so that any added func pointers are nullptr for forwards compatibility
    memset(&this->version, 0, sizeof(OrtCustomOp));

    int act_ver = GetActiveOrtAPIVersion();
    OrtCustomOp::version = act_ver < ORT_API_VERSION ? act_ver : ORT_API_VERSION;

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

#if ORT_API_VERSION >= 14
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp* op, size_t) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return (self->input_types_.empty() || self->input_types_[0] != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_VARIADIC;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* op, size_t) {
      auto self = reinterpret_cast<const OrtLiteCustomOp*>(op);
      return (self->output_types_.empty() || self->output_types_[0] != ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) ? INPUT_OUTPUT_OPTIONAL : INPUT_OUTPUT_VARIADIC;
    };

    OrtCustomOp::GetVariadicInputMinArity = [](const OrtCustomOp*) {
      return 1;
    };

    OrtCustomOp::GetVariadicInputHomogeneity = [](const OrtCustomOp*) {
      return 0;
    };

    OrtCustomOp::GetVariadicOutputMinArity = [](const OrtCustomOp*) {
      return 1;
    };

    OrtCustomOp::GetVariadicOutputHomogeneity = [](const OrtCustomOp*) {
      return 0;
    };

    OrtCustomOp::GetInputMemoryType = [](const OrtCustomOp*, size_t) {
      return OrtMemTypeDefault;
    };
#else
    OrtCustomOp::GetInputCharacteristic = [](const OrtCustomOp*, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };

    OrtCustomOp::GetOutputCharacteristic = [](const OrtCustomOp* op, size_t) {
      return INPUT_OUTPUT_OPTIONAL;
    };
#endif
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
  using CustomComputeFn = void (CustomOp::*)(Args...) const;
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

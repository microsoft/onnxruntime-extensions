// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_c_api.h"
#if ORT_API_VERSION >= 16

#include "onnxruntime_float16.h"
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif

namespace Ort {
namespace Custom {

// MFloat16
struct MFloat16 : onnxruntime_float16::Float16Impl<MFloat16> {
 private:
  constexpr explicit MFloat16(uint16_t v) noexcept { val = v; }

 public:
  using Base = onnxruntime_float16::Float16Impl<MFloat16>;

  MFloat16() = default;

  constexpr static MFloat16 FromBits(uint16_t v) noexcept { return MFloat16(v); }

  explicit MFloat16(float v) noexcept { val = Base::ToUint16Impl(v); }

  float ToFloat() const noexcept { return Base::ToFloatImpl(); }

  using Base::Abs;
  using Base::AreZero;
  using Base::IsFinite;
  using Base::IsInfinity;
  using Base::IsNaN;
  using Base::IsNaNOrZero;
  using Base::IsNegative;
  using Base::IsNegativeInfinity;
  using Base::IsNormal;
  using Base::IsPositiveInfinity;
  using Base::IsSubnormal;
  using Base::Negate;

  explicit operator float() const noexcept { return ToFloat(); }

  using Base::operator==;
  using Base::operator!=;
  using Base::operator<;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
#define ORTC_HOST_DEVICE __host__ __device__
#else
#define ORTC_HOST_DEVICE
#endif

// BFloat16
struct BFloat16 : onnxruntime_float16::BFloat16Impl<BFloat16> {
  using Base = onnxruntime_float16::BFloat16Impl<BFloat16>;

#if defined(__HIP__)
  ORTC_HOST_DEVICE BFloat16() = default;
#else
  BFloat16() = default;
#endif

  struct FromBitsT {};
  static constexpr ORTC_HOST_DEVICE FromBitsT FromBits() noexcept { return FromBitsT(); }
  constexpr ORTC_HOST_DEVICE BFloat16(unsigned short bits, FromBitsT) noexcept { val = bits; }

  static constexpr ORTC_HOST_DEVICE BFloat16 FromBits(uint16_t bits) noexcept {
    return BFloat16(bits, FromBits());
  }

  inline ORTC_HOST_DEVICE BFloat16(float v) noexcept {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    val = __bfloat16_as_ushort(__float2bfloat16(v));
#elif defined(__HIP__)
    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
    if (v != v) {  // isnan
      val = UINT16_C(0x7FC0);
    } else {
      union {
        uint32_t U32;
        float F32;
      };

      F32 = v;
      uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
      val = static_cast<uint16_t>((U32 + rounding_bias) >> 16);
    }
#else

    // Use C isnan to work both in host and device
    if (std::isnan(v)) {
      val = kPositiveQNaNBits;
    } else {
      auto get_msb_half = [](float fl) {
        uint16_t result;
        if constexpr (onnxruntime_float16::detail::endian::native == onnxruntime_float16::detail::endian::little) {
          std::memcpy(&result, reinterpret_cast<char*>(&fl) + sizeof(uint16_t), sizeof(uint16_t));
        } else {
          std::memcpy(&result, &fl, sizeof(uint16_t));
        }
        return result;
      };

      uint16_t upper_bits = get_msb_half(v);
      union {
        uint32_t U32;
        float F32;
      };
      F32 = v;
      U32 += (upper_bits & 1) + kRoundToNearest;
      val = get_msb_half(F32);
    }
#endif
  }

  inline ORTC_HOST_DEVICE float ToFloat() const noexcept {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(&val));
#elif defined(__HIP__)
    // We should be using memcpy in order to respect the strict aliasing rule but it fails in the HIP environment.
    float result = 0;
    uint32_t tmp = val;
    tmp <<= 16;
    float* tempRes = reinterpret_cast<float*>(&tmp);
    result = *tempRes;
    return result;
#else

    if (IsNaNHostDevice()) {
      return std::numeric_limits<float>::quiet_NaN();
    }

    float result = 0;
    char* const first = reinterpret_cast<char*>(&result);
    if constexpr (onnxruntime_float16::detail::endian::native == onnxruntime_float16::detail::endian::little) {
      char* const second = first + sizeof(uint16_t);
      std::memcpy(second, &val, sizeof(uint16_t));
    } else {
      std::memcpy(first, &val, sizeof(uint16_t));
    }
    return result;
#endif
  }

  static const BFloat16 NaN;
  static const BFloat16 NegativeNaN;
  static const BFloat16 Infinity;
  static const BFloat16 NegativeInfinity;
  static const BFloat16 Epsilon;
  static const BFloat16 MinValue;
  static const BFloat16 MaxValue;
  static const BFloat16 Zero;
  static const BFloat16 One;
  static const BFloat16 MinusOne;

  using Base::IsNegative;

  using Base::IsNaN;

  using Base::IsFinite;

  using Base::IsPositiveInfinity;

  using Base::IsNegativeInfinity;

  using Base::IsInfinity;

  using Base::IsNaNOrZero;

  using Base::IsNormal;

  using Base::IsSubnormal;

  using Base::Abs;

  using Base::Negate;

  ORTC_HOST_DEVICE operator float() const noexcept { return ToFloat(); }

#if defined(CUDA_VERSION) && CUDA_VERSION >= 11000
  ORTC_HOST_DEVICE BFloat16(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); }
  explicit ORTC_HOST_DEVICE operator __nv_bfloat16() const { return *reinterpret_cast<const __nv_bfloat16*>(&val); }
#endif

  ORTC_HOST_DEVICE bool operator==(const BFloat16& rhs) const noexcept {
    if (IsNaNHostDevice() || rhs.IsNaNHostDevice()) {
      // IEEE defines that NaN is not equal to anything, including itself.
      return false;
    }
    return val == rhs.val;
  }

  ORTC_HOST_DEVICE bool operator!=(const BFloat16& rhs) const noexcept {
    return !(*this == rhs);
  }

  ORTC_HOST_DEVICE bool operator<(const BFloat16& rhs) const noexcept {
    if (IsNaNHostDevice() || rhs.IsNaNHostDevice()) {
      // IEEE defines that NaN is unordered with respect to everything, including itself.
      return false;
    }

    const bool left_is_negative = IsNegativeHostDevice();
    if (left_is_negative != rhs.IsNegativeHostDevice()) {
      // When the signs of left and right differ, we know that left is less than right if it is
      // the negative value. The exception to this is if both values are zero, in which case IEEE
      // says they should be equal, even if the signs differ.
      return left_is_negative && !AreZeroHostDevice(*this, rhs);
    }
    return (val != rhs.val) && ((val < rhs.val) ^ left_is_negative);
  }

  ORTC_HOST_DEVICE bool IsNegativeHostDevice() const noexcept {
    return (val & kSignMask) != 0;
  }

  ORTC_HOST_DEVICE bool IsNaNHostDevice() const noexcept {
    return static_cast<uint16_t>(val & ~kSignMask) > kPositiveInfinityBits;
  }

  ORTC_HOST_DEVICE static bool AreZeroHostDevice(const BFloat16Impl& lhs, const BFloat16Impl& rhs) noexcept {
    // IEEE defines that positive and negative zero are equal, this gives us a quick equality check
    // for two values by or'ing the private bits together and stripping the sign. They are both zero,
    // and therefore equivalent, if the resulting value is still zero.
    return static_cast<uint16_t>((lhs.val | rhs.val) & ~kSignMask) == 0;
  }
};

}  // namespace Custom
}  // namespace Ort

#endif
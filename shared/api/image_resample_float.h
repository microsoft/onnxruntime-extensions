// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Float-domain bicubic resize with antialias for 3-channel (RGB) images.
//
// Matches torchvision F.resize(BICUBIC, antialias=True) / Pillow "F" mode:
//   same Keys cubic kernel (a=-0.5), same coefficient precomputation,
//   same separable horizontal→vertical pass order, NO uint8 clamping
//   (overshoot allowed).
//
// This is a shared utility — any image transform that needs float-domain
// bicubic resize can include this header instead of duplicating the code.
// Currently used by Gemma4; available for phi3, phi4, mllama, etc. if they
// need to match torchvision behavior precisely.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>
#include <algorithm>

namespace ort_extensions {
namespace detail {

inline double bicubic_kernel(double x) {
  // Keys cubic, a = -0.5
  if (x < 0.0) x = -x;
  if (x < 1.0) return ((1.5 * x - 2.5) * x) * x + 1.0;
  if (x < 2.0) return (((-0.5 * x + 2.5) * x - 4.0) * x + 2.0);
  return 0.0;
}

// Precompute filter coefficients for one dimension.
// Returns kernel size; fills bounds (x_start, count per output pixel)
// and coefficients (normalised weights).  Mirrors Pillow's precompute_coeffs().
inline int precompute_coeffs(
    int input_size, double input_start, double input_end, int output_size,
    std::vector<int>& bounds, std::vector<double>& coefficients) {
  constexpr double kSupport = 2.0;  // bicubic support

  double scale = (input_end - input_start) / output_size;
  double filter_scale = std::max(scale, 1.0);
  double support = kSupport * filter_scale;
  int kernel_size = static_cast<int>(std::ceil(support)) * 2 + 1;

  bounds.resize(static_cast<size_t>(output_size) * 2);
  coefficients.resize(static_cast<size_t>(output_size) * kernel_size);

  double inv_filter_scale = 1.0 / filter_scale;
  for (int output_x = 0; output_x < output_size; ++output_x) {
    double center = input_start + (output_x + 0.5) * scale;
    int x_start = static_cast<int>(center - support + 0.5);
    if (x_start < 0) x_start = 0;
    int x_count = static_cast<int>(center + support + 0.5);
    if (x_count > input_size) x_count = input_size;
    x_count -= x_start;

    double* coeffs = &coefficients[static_cast<size_t>(output_x) * kernel_size];
    double weight_sum = 0.0;
    for (int x = 0; x < x_count; ++x) {
      double weight = bicubic_kernel((x + x_start - center + 0.5) * inv_filter_scale);
      coeffs[x] = weight;
      weight_sum += weight;
    }
    if (weight_sum != 0.0) {
      for (int x = 0; x < x_count; ++x) coeffs[x] /= weight_sum;
    }
    for (int x = x_count; x < kernel_size; ++x) coeffs[x] = 0.0;
    bounds[static_cast<size_t>(output_x) * 2 + 0] = x_start;
    bounds[static_cast<size_t>(output_x) * 2 + 1] = x_count;
  }
  return kernel_size;
}

// Horizontal resample from uint8 source: (input_height, input_width, 3) uint8
// → (input_height, output_width, 3) float.
// Fuses the uint8→float rescale (1/255) into the filter accumulation to avoid
// allocating a full-resolution float copy of the source image.
inline void resample_horizontal_rgb_u8(
    float* destination, const uint8_t* source,
    int input_height, int input_width, int output_width,
    int kernel_size, const int* bounds, const double* coefficients) {
  constexpr double kRescale = 1.0 / 255.0;
  for (int y = 0; y < input_height; ++y) {
    const uint8_t* source_row = source + static_cast<size_t>(y) * input_width * 3;
    float* destination_row = destination + static_cast<size_t>(y) * output_width * 3;
    for (int output_x = 0; output_x < output_width; ++output_x) {
      int x_start = bounds[output_x * 2];
      int x_count = bounds[output_x * 2 + 1];
      const double* coeffs = &coefficients[static_cast<size_t>(output_x) * kernel_size];
      double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
      for (int x = 0; x < x_count; ++x) {
        const uint8_t* pixel = source_row + (x_start + x) * 3;
        sum_r += pixel[0] * coeffs[x];
        sum_g += pixel[1] * coeffs[x];
        sum_b += pixel[2] * coeffs[x];
      }
      destination_row[output_x * 3 + 0] = static_cast<float>(sum_r * kRescale);
      destination_row[output_x * 3 + 1] = static_cast<float>(sum_g * kRescale);
      destination_row[output_x * 3 + 2] = static_cast<float>(sum_b * kRescale);
    }
  }
}

// Horizontal resample: (input_height, input_width, 3) → (input_height, output_width, 3)
inline void resample_horizontal_rgb(
    float* destination, const float* source,
    int input_height, int input_width, int output_width,
    int kernel_size, const int* bounds, const double* coefficients) {
  for (int y = 0; y < input_height; ++y) {
    const float* source_row = source + static_cast<size_t>(y) * input_width * 3;
    float* destination_row = destination + static_cast<size_t>(y) * output_width * 3;
    for (int output_x = 0; output_x < output_width; ++output_x) {
      int x_start = bounds[output_x * 2];
      int x_count = bounds[output_x * 2 + 1];
      const double* coeffs = &coefficients[static_cast<size_t>(output_x) * kernel_size];
      double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
      for (int x = 0; x < x_count; ++x) {
        const float* pixel = source_row + (x_start + x) * 3;
        sum_r += pixel[0] * coeffs[x];
        sum_g += pixel[1] * coeffs[x];
        sum_b += pixel[2] * coeffs[x];
      }
      destination_row[output_x * 3 + 0] = static_cast<float>(sum_r);
      destination_row[output_x * 3 + 1] = static_cast<float>(sum_g);
      destination_row[output_x * 3 + 2] = static_cast<float>(sum_b);
    }
  }
}

// Vertical resample: (input_height, width, 3) → (output_height, width, 3)
inline void resample_vertical_rgb(
    float* destination, const float* source,
    int input_height, int width, int output_height,
    int kernel_size, const int* bounds, const double* coefficients) {
  for (int output_y = 0; output_y < output_height; ++output_y) {
    int y_start = bounds[output_y * 2];
    int y_count = bounds[output_y * 2 + 1];
    const double* coeffs = &coefficients[static_cast<size_t>(output_y) * kernel_size];
    float* destination_row = destination + static_cast<size_t>(output_y) * width * 3;
    for (int output_x = 0; output_x < width; ++output_x) {
      double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
      for (int y = 0; y < y_count; ++y) {
        const float* pixel = source + (static_cast<size_t>(y_start + y) * width + output_x) * 3;
        sum_r += pixel[0] * coeffs[y];
        sum_g += pixel[1] * coeffs[y];
        sum_b += pixel[2] * coeffs[y];
      }
      destination_row[output_x * 3 + 0] = static_cast<float>(sum_r);
      destination_row[output_x * 3 + 1] = static_cast<float>(sum_g);
      destination_row[output_x * 3 + 2] = static_cast<float>(sum_b);
    }
  }
}

}  // namespace detail

// Bicubic resize from uint8 RGB to float RGB:
// (source_height, source_width, 3) uint8 → (target_height, target_width, 3) float.
// Fuses the 1/255 rescale into the horizontal pass to avoid a full-resolution
// float copy of the source image. Output may exceed [0,1] due to bicubic
// overshoot (matching torchvision F.resize behavior).
//
// Parameters:
//   destination  — caller-allocated buffer of size
//                  target_height * target_width * 3 floats.  Output is
//                  HWC-interleaved float RGB, nominally in [0,1] but values
//                  may slightly exceed this range due to bicubic overshoot.
//   source       — source image in HWC uint8 RGB layout
//                  (source_height * source_width * 3 bytes).
inline void BicubicResizeU8ToFloatRGB(
    float* destination,
    const uint8_t* source, int source_height, int source_width,
    int target_height, int target_width) {
  // Precompute coefficients for each dimension
  std::vector<int> horizontal_bounds, vertical_bounds;
  std::vector<double> horizontal_coefficients, vertical_coefficients;
  int horizontal_kernel_size = detail::precompute_coeffs(
      source_width, 0.0, static_cast<double>(source_width), target_width,
      horizontal_bounds, horizontal_coefficients);
  int vertical_kernel_size = detail::precompute_coeffs(
      source_height, 0.0, static_cast<double>(source_height), target_height,
      vertical_bounds, vertical_coefficients);

  bool needs_horizontal = (target_width != source_width);
  bool needs_vertical = (target_height != source_height);

  if (needs_horizontal && needs_vertical) {
    // Horizontal pass reads uint8, writes float (fused rescale).
    // Only resample rows that the vertical pass will read.
    int ybox_first = vertical_bounds[0];
    int ybox_last = vertical_bounds[static_cast<size_t>(target_height - 1) * 2]
                  + vertical_bounds[static_cast<size_t>(target_height - 1) * 2 + 1];
    int temp_height = ybox_last - ybox_first;

    std::vector<float> temp(static_cast<size_t>(temp_height) * target_width * 3);
    detail::resample_horizontal_rgb_u8(
        temp.data(),
        source + static_cast<size_t>(ybox_first) * source_width * 3,
        temp_height, source_width, target_width,
        horizontal_kernel_size, horizontal_bounds.data(),
        horizontal_coefficients.data());

    // Shift vertical bounds to account for ybox_first offset
    for (int i = 0; i < target_height; ++i) {
      vertical_bounds[static_cast<size_t>(i) * 2] -= ybox_first;
    }

    // Vertical pass: (temp_height, target_width, 3) float
    //             → (target_height, target_width, 3) float
    detail::resample_vertical_rgb(
        destination, temp.data(), temp_height, target_width, target_height,
        vertical_kernel_size, vertical_bounds.data(),
        vertical_coefficients.data());
  } else if (needs_horizontal) {
    detail::resample_horizontal_rgb_u8(
        destination, source, source_height, source_width, target_width,
        horizontal_kernel_size, horizontal_bounds.data(),
        horizontal_coefficients.data());
  } else if (needs_vertical) {
    // Rare: same width, different height — convert to float first
    std::vector<float> source_float(
        static_cast<size_t>(source_height) * source_width * 3);
    constexpr float kRescale = 1.0f / 255.0f;
    for (size_t i = 0; i < source_float.size(); ++i)
      source_float[i] = static_cast<float>(source[i]) * kRescale;
    detail::resample_vertical_rgb(
        destination, source_float.data(), source_height, source_width,
        target_height, vertical_kernel_size, vertical_bounds.data(),
        vertical_coefficients.data());
  } else {
    constexpr float kRescale = 1.0f / 255.0f;
    for (size_t i = 0; i < static_cast<size_t>(source_height) * source_width * 3; ++i)
      destination[i] = static_cast<float>(source[i]) * kRescale;
  }
}

}  // namespace ort_extensions

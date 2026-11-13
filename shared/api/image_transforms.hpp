// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>

#include "ext_status.h"
#include "op_def_struct.h"
#include "image_resample.h"

template <typename T>
void DumpTensorToFile(const ortc::Tensor<T>& tensor, const char* name) {
#if _WIN32
  auto tic = GetTickCount();
  std::string dtype;
  if constexpr (std::is_same_v<T, uint8_t> || std::is_same_v<T, std::byte>) {
    dtype = "_u_";
  } else {
    dtype = "_f_";
  }
  dtype += std::to_string(tensor.Shape()[1]);
  // use tic to be filename in a temp file name
  auto filename = std::string("\\temp\\") + name + std::to_string(tic) + dtype + ".bin";
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  if (file.is_open()) {
    file.write(reinterpret_cast<const char*>(tensor.DataRaw()), tensor.SizeInBytes());
    file.close();
  }
#endif
}

template <typename T>
void SplitIntoTitles(const ortc::Tensor<T>& normalized_image,
                     ortc::Tensor<T>& pixel_values,
                     int64_t tile_height,
                     int64_t tile_width) {
  auto& shape = normalized_image.Shape();
  int64_t image_height = shape[0];
  int64_t image_width = shape[1];
  int64_t num_channels = shape[2];

  const int64_t image_1c_size = tile_height * tile_width;
  assert(image_height % tile_height == 0);
  int64_t num_tiles_height = static_cast<int64_t>(image_height / tile_height);
  assert(image_width % tile_width == 0);
  int64_t num_tiles_width = static_cast<int64_t>(image_width / tile_width);

  auto p_normalized_image = normalized_image.Data();
  // shape (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
  T* output_pixel =
      pixel_values.Allocate({num_tiles_height * num_tiles_width, num_channels, tile_height, tile_width});

  // From (image_height, image_width, num_channels)
  // Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
  for (int64_t i = 0; i < num_tiles_height; ++i) {
    for (int64_t j = 0; j < num_tiles_width; ++j) {
      // convert to be channel first
      for (int64_t k = 0; k < num_channels; ++k) {
        auto sub_index = image_1c_size * (i * num_tiles_width + j) * num_channels + image_1c_size * k;
        for (int64_t y = 0; y < tile_height; ++y) {
          for (int64_t x = 0; x < tile_width; ++x) {
            output_pixel[sub_index + y * tile_width + x] =
                p_normalized_image[(i * tile_height + y) * image_width * num_channels +
                                    (j * tile_width + x) * num_channels + k];
          }
        }
      }
    }
  }
}

inline OrtxStatus convert_to_rgb(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 3ULL || dimensions[2] != 3) {
    return {kOrtxErrorInvalidArgument, "[ConvertToRGB]: input is not (H, W, C)"};
  }

  std::uint8_t* p_output_image = output.Allocate(dimensions);
  auto* input_data = input.Data();
  auto h = dimensions[0];
  auto w = dimensions[1];
  auto c = dimensions[2];

  // convert BGR channel layouts to RGB
  for (int64_t j = 0; j < h; ++j) {
    for (int64_t k = 0; k < w; ++k) {
      auto c0_index = j * w * c + k * c;
      std::tie(p_output_image[c0_index], p_output_image[c0_index + 1], p_output_image[c0_index + 2]) =
          std::make_tuple(input_data[c0_index + 2], input_data[c0_index + 1], input_data[c0_index]);
    }
  }

  return {};
}

struct Resize {
  static const std::unordered_map<std::string, int>& InterpolationMethods() {
    static std::unordered_map<std::string, int> methods = {{"NEAREST", IMAGING_TRANSFORM_NEAREST},
                                                           {"LINEAR", IMAGING_TRANSFORM_BILINEAR},
                                                           {"CUBIC", IMAGING_TRANSFORM_BICUBIC},
                                                           {"CUBIC_OPENCV", IMAGING_TRANSFORM_BICUBIC},
                                                           {"LANCZOS", IMAGING_TRANSFORM_LANCZOS}};

    return methods;
  }

  int64_t round_by_factor(int64_t number, int64_t factor) {
      // Returns the closest integer to 'number' that is divisible by 'factor'.
      // Ties go to the even multiple, as Python's round() does in the Hugging Face smart_resize.
      int64_t q = number / factor, r = number % factor;
      if (2 * r > factor || (2 * r == factor && q % 2 != 0)) ++q;
      return q * factor;
  }

  int64_t ceil_by_factor(double number, int64_t factor) {
      // Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'.
      return static_cast<int64_t>(std::ceil(number / factor) * factor);
  }

  int64_t floor_by_factor(int64_t number, int64_t factor) {
      // Returns the largest integer less than or equal to 'number' that is divisible by 'factor'.
      return static_cast<int64_t>(std::floor(static_cast<double>(number) / factor) * factor);
  }

  std::tuple<int64_t, int64_t> smart_resize(int64_t height, int64_t width) {
      // Rescales the image to maintain the aspect ratio and adhere to pixel constraints.

      // Check if the aspect ratio exceeds the maximum allowed ratio.
      if (std::max(height, width) / static_cast<double>(std::min(height, width)) > max_ratio_) {
        throw std::invalid_argument("Absolute aspect ratio must be smaller than " + std::to_string(max_ratio_));
      }

      int64_t h_bar = std::max(static_cast<int64_t>(image_factor_), static_cast<int64_t>(round_by_factor(height, image_factor_)));
      int64_t w_bar = std::max(static_cast<int64_t>(image_factor_), static_cast<int64_t>(round_by_factor(width, image_factor_)));

      // Adjust the size if the pixel count is outside the min/max range.
      if (h_bar * w_bar > max_pixels_) {
          double beta = std::sqrt(static_cast<double>(height * width) / max_pixels_);
          h_bar = floor_by_factor(static_cast<int64_t>(height / beta), image_factor_);
          w_bar = floor_by_factor(static_cast<int64_t>(width / beta), image_factor_);
      } else if (h_bar * w_bar < min_pixels_) {
          double beta = std::sqrt(static_cast<double>(min_pixels_) / (height * width));
          h_bar = ceil_by_factor(height * beta, image_factor_);
          w_bar = ceil_by_factor(width * beta, image_factor_);
      }

      return std::make_tuple(h_bar, w_bar);
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Resize]: Only raw image formats"};
    }
    if (dimensions[2] != 3) {
      return {kOrtxErrorInvalidArgument,
              "[Resize]: Expected 3-channel RGB input, got " + std::to_string(dimensions[2]) + " channels"};
    }
    if (dimensions[0] <= 0 || dimensions[1] <= 0 || dimensions[0] > (std::numeric_limits<int>::max)() ||
        dimensions[1] > (std::numeric_limits<int>::max)() / 4 ||
        static_cast<uint64_t>(dimensions[0]) >
            (std::numeric_limits<size_t>::max)() / (static_cast<size_t>(dimensions[1]) * 3)) {
      return {kOrtxErrorInvalidArgument,
              "[Resize]: input geometry must fit positive imaging dimensions and RGB storage"};
    }
    const int source_height = static_cast<int>(dimensions[0]);
    const int source_width = static_cast<int>(dimensions[1]);
    int h = source_height, w = source_width;
    constexpr int c = 3;

    // Perform Smart Resize if Set
    // When smart_resize is enabled, use the original image dimensions (h, w)
    // to compute the target size that preserves the aspect ratio and snaps to
    // the patch grid, matching the HuggingFace smart_resize behavior.
    // When smart_resize is disabled, fall back to the fixed config dimensions.
    auto [height, width] = smart_resize_ ? smart_resize(static_cast<int64_t>(h), static_cast<int64_t>(w))
                                         : std::make_tuple(height_, width_);
    if (height <= 0 || width <= 0 || height > (std::numeric_limits<int>::max)() ||
        width > (std::numeric_limits<int>::max)() / 4) {
      return {kOrtxErrorInvalidArgument, "[Resize]: output geometry must fit positive imaging dimensions"};
    }
    h = static_cast<int>(height);
    w = static_cast<int>(width);
    
    if (keep_aspect_ratio_) {
      double scale = (std::max)(static_cast<double>(width) / w, static_cast<double>(height) / h);
      width = static_cast<int64_t>(w * scale);
      height = static_cast<int64_t>(h * scale);
    }

    if (height <= 0 || width <= 0 || height > (std::numeric_limits<int>::max)() ||
        width > (std::numeric_limits<int>::max)() / 4 ||
        static_cast<uint64_t>(height) > (std::numeric_limits<size_t>::max)() / (static_cast<size_t>(width) * 3)) {
      return {kOrtxErrorInvalidArgument, "[Resize]: output geometry must fit positive imaging dimensions"};
    }
    if (interpolation_ == "CUBIC_OPENCV") {
      ResizeOpenCVCubic(input.Data(), source_height, source_width, output.Allocate({height, width, c}),
                        static_cast<int>(height), static_cast<int>(width));
      return {};
    }
    using ImagingPtr = std::unique_ptr<std::remove_pointer_t<Imaging>, decltype(&ImagingDelete)>;
    ImagingPtr rgb_image(ImagingNew("RGB", source_width, source_height), ImagingDelete);
    if (!rgb_image) {
      return {kOrtxErrorOutOfMemory, "[Resize]: input imaging allocation failed"};
    }
    const auto* input_data = input.Data();
    for (int i = 0; i < source_height; ++i) {
      for (int j = 0; j < source_width; ++j) {
        uint8_t* pixel = reinterpret_cast<uint8_t*>(rgb_image->image[i] + static_cast<size_t>(j) * 4);
        const size_t offset = (static_cast<size_t>(i) * source_width + j) * 3;
        pixel[0] = input_data[offset];
        pixel[1] = input_data[offset + 1];
        pixel[2] = input_data[offset + 2];
        pixel[3] = 0;
      }
    }
    const int interp = InterpolationMethods().at(interpolation_);
    float box[4] = {0.0f, 0.0f, static_cast<float>(source_width), static_cast<float>(source_height)};
    ImagingPtr output_image(
        int16_coeffs_
            ? ImagingResampleInt16(rgb_image.get(), static_cast<int>(width), static_cast<int>(height), interp, box)
            : ImagingResample(rgb_image.get(), static_cast<int>(width), static_cast<int>(height), interp, box),
        ImagingDelete);
    rgb_image.reset();
    if (!output_image) {
      return {kOrtxErrorInvalidArgument, "[Resize]: ImagingResample failed (invalid parameters, unsupported mode, or out of memory)"};
    }

    auto* p_output_image = output.Allocate({height, width, c});
    for (auto i = height - height; i < height; ++i) {
      for (auto j = width - width; j < width; ++j) {
      auto c0_index = i * width * c + j * c;
      std::memcpy(p_output_image + c0_index, output_image->image[i] + j * 4, c);
      }
    }

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "height") {
        height_ = std::get<int64_t>(value);
      } else if (key == "width") {
        width_ = std::get<int64_t>(value);
      } else if (key == "keep_aspect_ratio") {
        keep_aspect_ratio_ = std::get<int64_t>(value) != 0;
      } else if (key == "interpolation") {
        interpolation_ = std::get<std::string>(value);
        if (InterpolationMethods().find(interpolation_) == InterpolationMethods().end()) {
          return {kOrtxErrorInvalidArgument, "[Resize]: Invalid interpolation method"};
        }
      } else if (key == "coefficient_type") {
        const auto& type = std::get<std::string>(value);
        if (type != "int16" && type != "int32") {
          return {kOrtxErrorInvalidArgument, "[Resize]: coefficient_type must be int16 or int32"};
        }
        int16_coeffs_ = type == "int16";
      } else if (key == "smart_resize") {
        smart_resize_ = std::get<int64_t>(value) != 0;
      } else if (key == "min_pixels") {
        min_pixels_ = std::get<int64_t>(value);
      } else if (key == "max_pixels") {
        max_pixels_ = std::get<int64_t>(value);
      } else if (key == "patch_size") {
        patch_size_ = std::get<int64_t>(value);
      } else if (key == "merge_size") {
        merge_size_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Resize]: Invalid argument"};
      }
    }
    if (patch_size_ <= 0 || merge_size_ <= 0) {
      return {kOrtxErrorInvalidArgument, "[Resize]: patch_size and merge_size must be positive"};
    }
    image_factor_ = patch_size_ * merge_size_;
    return {};
  }

 private:
  void ResizeOpenCVCubic(const uint8_t* source, int source_height, int source_width, uint8_t* destination,
                         int target_height, int target_width) {
    // Four-tap, half-pixel cubic without antialiasing. Clip and round only after both passes.
    auto coefficients = [](double fraction) {
      constexpr double a = -0.75;
      const double next = fraction + 1.0, other = 1.0 - fraction;
      std::array<double, 4> weights;
      weights[0] = ((a * next - 5.0 * a) * next + 8.0 * a) * next - 4.0 * a;
      weights[1] = ((a + 2.0) * fraction - (a + 3.0)) * fraction * fraction + 1.0;
      weights[2] = ((a + 2.0) * other - (a + 3.0)) * other * other + 1.0;
      weights[3] = 1.0 - weights[0] - weights[1] - weights[2];
      return weights;
    };
    const double scale_y = static_cast<double>(source_height) / target_height;
    const double scale_x = static_cast<double>(source_width) / target_width;
    for (int y = 0; y < target_height; ++y) {
      const double position_y = (y + 0.5) * scale_y - 0.5;
      const int64_t base_y = static_cast<int64_t>(std::floor(position_y));
      const auto vertical = coefficients(position_y - static_cast<double>(base_y));
      for (int x = 0; x < target_width; ++x) {
        const double position_x = (x + 0.5) * scale_x - 0.5;
        const int64_t base_x = static_cast<int64_t>(std::floor(position_x));
        const auto horizontal = coefficients(position_x - static_cast<double>(base_x));
        for (size_t channel = 0; channel < 3; ++channel) {
          double value = 0.0;
          for (int row = 0; row < 4; ++row) {
            const auto source_y = static_cast<size_t>(std::clamp<int64_t>(base_y + row - 1, 0, source_height - 1));
            double line = 0.0;
            for (int column = 0; column < 4; ++column) {
              const auto source_x = static_cast<size_t>(std::clamp<int64_t>(base_x + column - 1, 0, source_width - 1));
              line += source[(source_y * source_width + source_x) * 3 + channel] * horizontal[column];
            }
            value += line * vertical[row];
          }
          value = std::clamp(value, 0.0, 255.0);
          int rounded = static_cast<int>(value);
          const double remainder = value - rounded;
          if (remainder > 0.5 || (remainder == 0.5 && rounded % 2 != 0)) ++rounded;
          destination[(static_cast<size_t>(y) * target_width + x) * 3 + channel] = static_cast<uint8_t>(rounded);
        }
      }
    }
  }

  int64_t height_{256};
  int64_t width_{256};
  bool keep_aspect_ratio_{true};
  std::string interpolation_{"CUBIC"};  // LINEAR, NEAREST, CUBIC

  /*
  These constants below are defaults from Qwen2/Qwen2.5-VL image processing, and come from
  the config: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/blob/main/preprocessor_config.json#L2

  Some have calculations behind them, for instance:
  min_pixels_ = 4 * 28 * 28
  max_pixels_ = 16384 * 28 * 28,
  but they are passed in through a JSON config and hence passed as constants (same for defaults).
  */

  bool smart_resize_{false};
  bool int16_coeffs_{false};
  int64_t image_factor_{28};
  int64_t min_pixels_{3136};
  int64_t max_pixels_{12845056};
  int64_t patch_size_{14};
  int64_t merge_size_{2};
  double max_ratio_{200.0};
};

struct Rescale {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "rescale_factor") {
        scale_ = static_cast<float>(std::get<double>(value));
      } else {
        return {kOrtxErrorInvalidArgument, "[Rescale]: Invalid argument"};
      }
    }

    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<float>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {  // Only raw image formats
      return {kOrtxErrorInvalidArgument, "[Rescale]: Only raw image formats"};
    }

    auto* input_data = input.Data();
    auto h = dimensions[0];
    auto w = dimensions[1];
    auto c = dimensions[2];
    auto* p_output_image = output.Allocate({h, w, c});

    for (int64_t j = 0; j < h; ++j) {
      for (int64_t k = 0; k < w; ++k) {
        auto c0_index = j * w * c + k * c;
        for (int64_t l = 0; l < c; ++l) {
          p_output_image[c0_index + l] = static_cast<float>(input_data[c0_index + l]) * scale_;
        }
      }
    }

    return {};
  }

 private:
  float scale_{1.0f / 255.0f};
};

struct Normalize {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "mean") {
        auto mean = std::get<std::vector<double>>(value);
        if (mean.size() < 3) {
          return {kOrtxErrorInvalidArgument, "[Normalize]: mean must have at least 3 values"};
        }
        mean_ = {static_cast<float>(mean[0]), static_cast<float>(mean[1]), static_cast<float>(mean[2])};
      } else if (key == "std") {
        auto std = std::get<std::vector<double>>(value);
        if (std.size() < 3) {
          return {kOrtxErrorInvalidArgument, "[Normalize]: std must have at least 3 values"};
        }
        std_ = {static_cast<float>(std[0]), static_cast<float>(std[1]), static_cast<float>(std[2])};
      } else if (key == "qwen2_5_vl" || key == "qwen3_vl") {
        qwen2_5_vl_ = std::get<int64_t>(value) != 0;
      } else {
        return {kOrtxErrorInvalidArgument, "[Normalize]: Invalid argument"};
      }
    }
    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output) {
    const auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Normalize]: Only raw image formats"};
    }

    const float* input_data = input.Data();
    int64_t H = dimensions[0];
    int64_t W = dimensions[1];
    int64_t C = dimensions[2];

    if (C != static_cast<int64_t>(mean_.size()) || C != static_cast<int64_t>(std_.size())) {
      return {kOrtxErrorInvalidArgument,
              "[Normalize]: channel count " + std::to_string(C) +
              " does not match mean/std size (mean=" + std::to_string(mean_.size()) +
              ", std=" + std::to_string(std_.size()) + ")"};
    }

    float* out = output.Allocate({H, W, C});

    if (!qwen2_5_vl_) {
      // Default: HWC direct normalization
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t idx = (h * W + w) * C;
          for (int c = 0; c < C; c++) {
            out[idx + c] = (input_data[idx + c] - mean_[c]) / std_[c];
          }
        }
      }
    } else {
      // Qwen2.5-VL: Swap BGR -> RGB before normalization
      for (int64_t h = 0; h < H; h++) {
        for (int64_t w = 0; w < W; w++) {
          size_t idx = (h * W + w) * C;

          float B = input_data[idx + 0];
          float G = input_data[idx + 1];
          float R = input_data[idx + 2];

          // Normalize in RGB order
          out[idx + 0] = (R - mean_[0]) / std_[0];
          out[idx + 1] = (G - mean_[1]) / std_[1];
          out[idx + 2] = (B - mean_[2]) / std_[2];
        }
      }
    }

    return {};
  }

 private:
  // Constants set to defaults
  std::vector<float> mean_{0.48145466f, 0.4578275f, 0.40821073f};
  std::vector<float> std_{0.26862954f, 0.26130258f, 0.27577711f};
  bool qwen2_5_vl_{false};
};

struct CenterCrop {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "height") {
        target_h_ = std::get<int64_t>(value);
      } else if (key == "width") {
        target_w_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[CenterCrop]: Invalid attribute " + key};
      }
    }

    return {};
  }

  //   # T.CenterCrop(224),
  // width, height = self.target_size, self.target_size
  // img_h, img_w = img.shape[-2:]
  // s_h = torch.div((img_h - height), 2, rounding_mode='trunc')
  // s_w = torch.div((img_w - width), 2, rounding_mode='trunc')
  // x = img[:, :, s_h:s_h + height, s_w:s_w + width]
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[CenterCrop]: Only raw image formats"};
    }

    auto* input_data = input.Data();
    auto h = dimensions[0];
    auto w = dimensions[1];
    auto c = dimensions[2];

    if (target_h_ > h || target_w_ > w) {
      return {kOrtxErrorInvalidArgument, "[CenterCrop]: target dimensions exceed input image size"};
    }

    auto* p_output_image = output.Allocate({target_h_, target_w_, c});
    auto s_h = (h - target_h_) / 2;
    auto s_w = (w - target_w_) / 2;

    for (int64_t j = 0; j < target_h_; ++j) {
      for (int64_t k = 0; k < target_w_; ++k) {
        auto c0_index = (j + s_h) * w * c + (k + s_w) * c;
        for (int64_t l = 0; l < c; ++l) {
          p_output_image[j * target_w_ * c + k * c + l] = input_data[c0_index + l];
        }
      }
    }

    return {};
  }

 private:
  int64_t target_h_{224};
  int64_t target_w_{224};
};

struct Permute3D {

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL || dims_.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Permute]: Only 3D tensors are supported"};
    }

    auto* input_data = input.Data();
    std::vector<int64_t> output_shape = {dimensions[dims_[0]], dimensions[dims_[1]], dimensions[dims_[2]]};
    auto* p_output_image = output.Allocate(output_shape);

    for (int64_t i = 0; i < dimensions[0]; ++i) {
      for (int64_t j = 0; j < dimensions[1]; ++j) {
        for (int64_t k = 0; k < dimensions[2]; ++k) {
          auto c0_index = i * dimensions[1] * dimensions[2] + j * dimensions[2] + k;
          auto c1_index = (dims_[0] == 0 ? i : (dims_[0] == 1 ? j : k)) * output_shape[1] * output_shape[2] +
                          (dims_[1] == 0 ? i : (dims_[1] == 1 ? j : k)) * output_shape[2] +
                          (dims_[2] == 0 ? i : (dims_[2] == 1 ? j : k));
          p_output_image[c1_index] = input_data[c0_index];
        }
      }
    }

    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "dims") {
        dims_ = std::get<std::vector<int64_t>>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Permute]: Invalid argument"};
      }
    }

    if (dims_.size() != 3 || dims_[0] < 0 || dims_[0] > 2 ||
        dims_[1] < 0 || dims_[1] > 2 || dims_[2] < 0 || dims_[2] > 2 ||
        dims_[0] == dims_[1] || dims_[0] == dims_[2] || dims_[1] == dims_[2]) {
      return {kOrtxErrorInvalidArgument, "[Permute]: dims must be a permutation of {0, 1, 2}"};
    }

    return {};
  }

 private:
  std::vector<int64_t> dims_{1, 2, 0};
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
    static std::unordered_map<std::string, int> methods = {
      {"NEAREST", IMAGING_TRANSFORM_NEAREST},
      {"LINEAR", IMAGING_TRANSFORM_BILINEAR},
      {"CUBIC", IMAGING_TRANSFORM_BICUBIC},
      {"LANCZOS", IMAGING_TRANSFORM_LANCZOS}
    };

    return methods;
  }

  int64_t round_by_factor(int64_t number, int64_t factor) {
      // Returns the closest integer to 'number' that is divisible by 'factor'.
      return static_cast<int>(std::round(static_cast<double>(number) / factor) * factor);
  }

  int64_t ceil_by_factor(int64_t number, int64_t factor) {
      // Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'.
      return static_cast<int>(std::ceil(static_cast<double>(number) / factor) * factor);
  }

  int64_t floor_by_factor(int64_t number, int64_t factor) {
      // Returns the largest integer less than or equal to 'number' that is divisible by 'factor'.
      return static_cast<int>(std::floor(static_cast<double>(number) / factor) * factor);
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
          h_bar = ceil_by_factor(static_cast<int64_t>(height * beta), image_factor_);
          w_bar = ceil_by_factor(static_cast<int64_t>(width * beta), image_factor_);
      }

      return std::make_tuple(h_bar, w_bar);
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input, ortc::Tensor<uint8_t>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Resize]: Only raw image formats"};
    }

    auto* input_data = input.Data();
    int h = static_cast<int>(dimensions[0]);
    int w = static_cast<int>(dimensions[1]);
    int c = static_cast<int>(dimensions[2]);

    Imaging rgb_image = ImagingNew("RGB", w, h);
    for (int32_t i = 0; i < h; ++i) {
      for (int32_t j = 0; j < w; ++j) {
        uint8_t* pixel = reinterpret_cast<uint8_t*>(rgb_image->image[i] + j * 4);
        pixel[0] = input_data[(i * w + j) * 3];
        pixel[1] = input_data[(i * w + j) * 3 + 1];
        pixel[2] = input_data[(i * w + j) * 3 + 2];
        pixel[3] = 0;  // unused
      }
    }

    int interp = InterpolationMethods().at(interpolation_);
    float box[4] = {0.0f, 0.0f, static_cast<float>(w), static_cast<float>(h)};

    image_factor_ = patch_size_ * merge_size_;

    // Perform Smart Resize if Set
    auto [height, width] = smart_resize_ ? smart_resize(height_, width_) : std::make_tuple(height_, width_);
    h = static_cast<int>(height);
    w = static_cast<int>(width);
    
    if (keep_aspect_ratio_) {
      double scale = (std::max)(static_cast<double>(width) / w, static_cast<double>(height) / h);
      width = static_cast<int64_t>(w * scale);
      height = static_cast<int64_t>(h * scale);
    }

    auto output_image = ImagingResample(rgb_image, static_cast<int>(width), static_cast<int>(height), interp, box);
    ImagingDelete(rgb_image);

    auto* p_output_image = output.Allocate({height, width, c});
    for (auto i = height - height; i < height; ++i) {
      for (auto j = width - width; j < width; ++j) {
      auto c0_index = i * width * c + j * c;
      std::memcpy(p_output_image + c0_index, output_image->image[i] + j * 4, c);
      }
    }

    ImagingDelete(output_image);
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
    return {};
  }

 private:
  int64_t height_{256};
  int64_t width_{256};
  bool keep_aspect_ratio_{true};
  std::string interpolation_{"CUBIC"};  // LINEAR, NEAREST, CUBIC

  bool smart_resize_{false};
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
        mean_ = {static_cast<float>(mean[0]), static_cast<float>(mean[1]), static_cast<float>(mean[2])};
      } else if (key == "std") {
        auto std = std::get<std::vector<double>>(value);
        std_ = {static_cast<float>(std[0]), static_cast<float>(std[1]), static_cast<float>(std[2])};
      } else if (key == "qwen2_5_vl") {
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

    return {};
  }

 private:
  std::vector<int64_t> dims_{1, 2, 0};
};

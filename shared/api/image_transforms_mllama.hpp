// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "ortx_processor.h"
#include "c_api_utils.hpp"
#include "image_resample.h"
#include "image_transforms.hpp"

struct Llama3ImageTransform {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    DictT normalizer_attrs;
    DictT rescaler_attrs;
    for (const auto& [key, value] : attrs) {
      if (key.find("normalize/") == 0) {
        normalizer_attrs[key.substr(10)] = value;
      } else if (key.find("rescale/") == 0) {
        rescaler_attrs[key.substr(8)] = value;
      } else if (key == "max_image_tiles") {
        max_image_tiles_ = std::get<int64_t>(value);
      } else if (key == "size") {
        auto tile_size = std::get<std::vector<int64_t>>(value);
        if (tile_size.size() != 2) {
          return {kOrtxErrorInvalidArgument, "[Llama3ImageTransform]: Invalid tile size"};
        }
        tile_size_ = std::make_pair(tile_size[0], tile_size[1]);
      } else if (key == "interpolation") {
        interpolation_ = std::get<std::string>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Llama3ImageTransform]: Invalid argument"};
      }
    }

    OrtxStatus status = normalize_.Init(normalizer_attrs);
    if (!status.IsOk()) {
      return status;
    }

    return rescale_.Init(rescaler_attrs);
  }

  void ProcessImageTiles(const ortc::Tensor<float>& normalized_image, ortc::Tensor<float>& pixel_values) {
    auto& shape = normalized_image.Shape();
    auto c = shape[2];
    auto p_pixel_values = normalized_image.Data();

    const auto image_1c_size = tile_size_.first * tile_size_.second;
    int m = static_cast<int>(shape[0] / tile_size_.first);
    int n = static_cast<int>(shape[1] / tile_size_.second);

    float* output_pixel = pixel_values.Allocate({m * n, c, tile_size_.first, tile_size_.second});
    // Permute to (num_tiles_height, num_tiles_width, num_channels, tile_height, tile_width)
    // then Reshape into the desired output shape
    //   (num_tiles_width * num_tiles_height, num_channels, tile_height, tile_width)
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int32_t k = 0; k < c; ++k) {
          // convert to be channel first
          auto sub_index = (static_cast<int64_t>(i) * n + j) * image_1c_size * c + k * image_1c_size;
          for (int y = 0; y < tile_size_.first; ++y) {
            for (int x = 0; x < tile_size_.second; ++x) {
              output_pixel[sub_index + y * tile_size_.second + x] =
                  p_pixel_values[k * shape[0] * shape[1] + (i * tile_size_.first + y) * shape[1] +
                                 (j * tile_size_.second + x)];
            }
          }
        }
      }
    }
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& image, ortc::Tensor<float>& pixel_values,
                     ortc::Tensor<int64_t>& aspect_ratio_ids, ortc::Tensor<int64_t>& aspect_ratio_mask,
                     ortc::Tensor<int64_t>& num_tiles) {
    auto& dimensions = image.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Llama3ImageTransform]: Only 3D decoded image tensors are supported"};
    }

    std::pair<int64_t, int64_t> aspect_ratio;
    ortc::Tensor<uint8_t> resized_image(&ortx::CppAllocator::Instance());
    OrtxStatus status = DoResize(image, resized_image, aspect_ratio);
    if (!status.IsOk()) {
      return status;
    }

    ortc::Tensor<uint8_t> padded_image(&ortx::CppAllocator::Instance());
    status = DoPad(resized_image, aspect_ratio, padded_image);
    if (!status.IsOk()) {
      return status;
    }

    ortc::Tensor<float> rescaled_image(&ortx::CppAllocator::Instance());
    status = rescale_.Compute(padded_image, rescaled_image);
    if (!status.IsOk()) {
      return status;
    }

    ortc::Tensor<float> normalized_image(&ortx::CppAllocator::Instance());
    status = normalize_.Compute(rescaled_image, normalized_image);
    if (!status.IsOk()) {
      return status;
    }

    ProcessImageTiles(normalized_image, pixel_values);

    std::vector<std::pair<int64_t, int64_t>> aspect_ratios = {aspect_ratio};
    auto v_aspect_ratio_ids = ConvertAspectRatiosToIds(aspect_ratios, max_image_tiles_);
    auto v_aspect_ratio_mask = BuildAspectRatioMask(aspect_ratios, max_image_tiles_);

    auto p_ids = aspect_ratio_ids.Allocate({static_cast<int64_t>(v_aspect_ratio_ids.size())});
    std::copy(v_aspect_ratio_ids.begin(), v_aspect_ratio_ids.end(), p_ids);

    auto p_mask = aspect_ratio_mask.Allocate({static_cast<int64_t>(v_aspect_ratio_mask[0].size())});
    std::copy(v_aspect_ratio_mask[0].begin(), v_aspect_ratio_mask[0].end(), p_mask);

    auto p_num_tiles = num_tiles.Allocate({1});
    p_num_tiles[0] = aspect_ratios[0].first * aspect_ratios[0].second;

    return status;
  }

 private:
  static std::vector<std::pair<int64_t, int64_t>> GetAllSupportedAspectRatios(int64_t max_image_tiles) {
    std::vector<std::pair<int64_t, int64_t>> aspect_ratios;

    for (int64_t width = 1; width <= max_image_tiles; ++width) {
      for (int64_t height = 1; height <= max_image_tiles; ++height) {
        if (width * height <= max_image_tiles) {
          aspect_ratios.emplace_back(width, height);
        }
      }
    }

    return aspect_ratios;
  }

  static std::tuple<int64_t, int64_t> GetImageSizeFitToCanvas(int64_t image_height, int64_t image_width,
                                                              int64_t canvas_height, int64_t canvas_width,
                                                              int64_t tile_size) {
    // Set target image size in between `tile_size` and canvas_size
    int64_t target_width = std::clamp(image_width, tile_size, canvas_width);
    int64_t target_height = std::clamp(image_height, tile_size, canvas_height);

    double scale_h = static_cast<double>(target_height) / image_height;
    double scale_w = static_cast<double>(target_width) / image_width;

    int64_t new_width, new_height;

    if (scale_w < scale_h) {
      new_width = target_width;
      new_height = std::min(static_cast<int64_t>(std::floor(image_height * scale_w)), target_height);
    } else {
      new_height = target_height;
      new_width = std::min(static_cast<int64_t>(std::floor(image_width * scale_h)), target_width);
    }

    return std::make_tuple(new_height, new_width);
  }

  static std::vector<std::vector<int64_t>> BuildAspectRatioMask(
      const std::vector<std::pair<int64_t, int64_t>>& aspect_ratios, int64_t max_image_tiles) {
    int64_t max_num_images = aspect_ratios.size();

    // Initialize the 2D vector with zeros
    std::vector<std::vector<int64_t>> aspect_ratio_mask(max_num_images, std::vector<int64_t>(max_image_tiles, 0));

    // Set the first tile to 1 for all aspect ratios
    for (int64_t j = 0; j < max_num_images; ++j) {
      aspect_ratio_mask[j][0] = 1;
    }

    // Set the aspect ratio mask for the rest of the tiles
    for (size_t j = 0; j < aspect_ratios.size(); ++j) {
      int64_t num_tiles_w = aspect_ratios[j].first;
      int64_t num_tiles_h = aspect_ratios[j].second;
      int64_t num_tiles = num_tiles_w * num_tiles_h;
      for (int64_t k = 0; k < num_tiles && k < max_image_tiles; ++k) {
        aspect_ratio_mask[j][k] = 1;
      }
    }

    return aspect_ratio_mask;
  }

  static std::pair<int64_t, int64_t> GetOptimalTiledCanvas(int64_t image_height, int64_t image_width,
                                                            int64_t max_image_tiles, int64_t tile_size) {
    auto possible_tile_arrangements = GetAllSupportedAspectRatios(max_image_tiles);
    std::vector<std::pair<int64_t, int64_t>> possible_canvas_sizes;

    for (const auto& arrangement : possible_tile_arrangements) {
      possible_canvas_sizes.emplace_back(arrangement.first * tile_size, arrangement.second * tile_size);
    }

    std::vector<double> scales;
    std::vector<std::pair<int64_t, int64_t>> chosen_canvas;
    double selected_scale;

    for (const auto& canvas : possible_canvas_sizes) {
      double scale_h = static_cast<double>(canvas.second) / image_height;
      double scale_w = static_cast<double>(canvas.first) / image_width;
      double scale = std::min(scale_h, scale_w);
      scales.push_back(scale);
    }

    auto upscaling_it = std::find_if(scales.begin(), scales.end(), [](double scale) { return scale >= 1.0; });

    if (upscaling_it != scales.end()) {
      selected_scale = *std::min_element(upscaling_it, scales.end());
    } else {
      selected_scale = *std::max_element(scales.begin(), scales.end());
    }

    for (size_t i = 0; i < scales.size(); ++i) {
      if (std::abs(scales[i] - selected_scale) < std::numeric_limits<double>::epsilon()) {
        chosen_canvas.push_back(possible_canvas_sizes[i]);
      }
    }

    std::pair<int64_t, int64_t> optimal_canvas;

    if (chosen_canvas.size() > 1) {
      int64_t min_area = std::numeric_limits<int64_t>::max();
      for (const auto& canvas : chosen_canvas) {
        int64_t area = canvas.first * canvas.second;
        if (area < min_area) {
          min_area = area;
          optimal_canvas = canvas;
        }
      }
    } else {
      optimal_canvas = chosen_canvas[0];
    }

    return std::make_pair(optimal_canvas.second, optimal_canvas.first);
  }

  static std::vector<int64_t> ConvertAspectRatiosToIds(const std::vector<std::pair<int64_t, int64_t>>& aspect_ratios,
                                                       int64_t max_image_tiles) {
    int64_t max_num_images = aspect_ratios.size();

    auto supported_aspect_ratios = GetAllSupportedAspectRatios(max_image_tiles);

    // Initialize the 1D vector with zeros
    std::vector<int64_t> aspect_ratios_ids(max_num_images, 0);

    for (size_t j = 0; j < aspect_ratios.size(); ++j) {
      const auto& ratio = aspect_ratios[j];
      auto it = std::find(supported_aspect_ratios.begin(), supported_aspect_ratios.end(), ratio);
      if (it != supported_aspect_ratios.end()) {
        aspect_ratios_ids[j] = std::distance(supported_aspect_ratios.begin(), it) + 1;
      }
    }

    return aspect_ratios_ids;
  }

  OrtxStatus DoPad(const ortc::Tensor<uint8_t>& image, const std::pair<int64_t, int64_t>& aspect_ratio,
                   ortc::Tensor<uint8_t>& padded_image) const{
    auto& dimensions = image.Shape();
    auto [image_height, image_width] = std::make_tuple(dimensions[0], dimensions[1]);
    auto [num_tiles_height, num_tiles_width] = aspect_ratio;
    auto padded_height = num_tiles_height * tile_size_.first;
    auto padded_width = num_tiles_width * tile_size_.second;
    auto pad_size = std::make_pair(padded_height - image_height, padded_width - image_width);
    auto channels = dimensions[2];
    auto* padded_image_data = padded_image.Allocate({padded_height, padded_width, channels});
    std::memset(padded_image_data, 0, padded_height * padded_width * channels);
    auto* input_data = image.Data();
    for (int64_t j = 0; j < image_height; ++j) {
      std::memcpy(padded_image_data + j * padded_width * channels, input_data + j * image_width * channels,
                  image_width * channels);
    }

    return {};
  }

  OrtxStatus DoResize(const ortc::Tensor<uint8_t>& image, ortc::Tensor<uint8_t>& resized_image,
                      std::pair<int64_t, int64_t>& aspect_ratio) const{
    auto& dimensions = image.Shape();
    auto [image_height, image_width] = std::make_tuple(dimensions[0], dimensions[1]);
    auto tile_size = tile_size_.first;
    auto [canvas_height, canvas_width] = GetOptimalTiledCanvas(image_height, image_width, max_image_tiles_, tile_size);
    auto num_tiles_height = canvas_height / tile_size;
    auto num_tiles_width = canvas_width / tile_size;
    aspect_ratio = std::make_pair(num_tiles_height, num_tiles_width);
    auto [new_height, new_width] =
        GetImageSizeFitToCanvas(image_height, image_width, canvas_height, canvas_width, tile_size);
    Resize resizer;
    std::unordered_map<std::string, ortx::AttrType> attrs = {
        {"height", new_height}, {"width", new_width}, {"interpolation", std::string("LINEAR")}};
    OrtxStatus status = resizer.Init(attrs);
    if (!status.IsOk()) {
      return status;
    }

    return resizer.Compute(image, resized_image);
  }

 private:
  int64_t max_image_tiles_{};
  std::pair<int64_t, int64_t> tile_size_{};
  std::string interpolation_{};

  Rescale rescale_;
  Normalize normalize_;
};

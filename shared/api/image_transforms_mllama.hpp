// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "ortx_processor.h"
#include "c_api_utils.hpp"
#include "image_resample.h"
#include "image_transforms.hpp"

struct Llama3ImageTransform {
  static void SplitIntoTitles(const ortc::Tensor<float>& normalized_image, ortc::Tensor<float>& pixel_values,
                              int64_t tile_height, int64_t tile_width) {
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
    float* output_pixel =
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
    resized_image.Release();

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

    // DumpTensorToFile(normalized_image, "normalized_image");

    SplitIntoTitles(normalized_image, pixel_values, tile_size_.first, tile_size_.second);

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

  /*
    Calculates the new size of an image to fit within a canvas while maintaining aspect ratio.

    This function calculates the optimal size for an image to fit within a canvas defined by
    canvas_height and canvas_width, while ensuring that the image dimensions are not smaller than
    tile_size. If the image is larger than the canvas, the returned size will fit within the canvas.
    If the image already fits within the canvas, the size remains unchanged.
    The aspect ratio of the original image is preserved.

    Args:
        image_height (`int`):
            The height of the original image.
        image_width (`int`):
            The width of the original image.
        canvas_height (`int`):
            The height of the canvas.
        canvas_width (`int`):
            The width of the canvas.
        tile_size (`int`):
            The tile size.

    Returns:
        `Tuple[int, int]`: A tuple containing the new height and width of the image.

  */
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
      new_height = static_cast<int64_t>(std::round(image_height * scale_w));
    } else {
      new_height = target_height;
      new_width = static_cast<int64_t>(std::round(image_width * scale_h));
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

  /*
  Determines the best canvas based on image and tile size and maximum number of tiles.

  First, calculates possible resolutions based on the maximum number of tiles and tile size.
  For example for max_image_tiles=2, tile_size=100, possible tile arrangements are:
  [(1, 1), (1, 2), (2, 1)] and corresponding canvas sizes are:
  [(100, 100), (100, 200), (200, 100)]

  For each possible resolution, calculates the scaling factors for
  width and height, and selects the smallest one, which is the limiting side.
  E.g. to match the canvas you can upscale height by 2x, and width by 1.5x,
  therefore, the maximum upscaling you can do is min(2, 1.5) = 1.5.

  If upscaling is possible (any of the scaling factors is greater than 1),
  then picks the smallest upscaling factor > 1.

  If upscaling is not possible, then picks the largest scaling factor <= 1, i.e.
  reduce downscaling as much as possible.

  If there are multiple resolutions with the same max scale, we pick the one with the lowest area,
  to minimize padding. E.g., the same image can be upscaled to 224x224 and 224x448, but the latter
  has more padding.

  Args:
      image_height (`int`):
          The height of the image.
      image_width (`int`):
          The width of the image.
      max_image_tiles (`int`):
          The maximum number of tiles any image can be split into.
      tile_size (`int`):
          The tile size.

  Returns:
      `pair[int, int]`: The best canvas resolution [height, width] for the given image.
  */
  static std::pair<int64_t, int64_t> GetOptimalTiledCanvas(int64_t image_height, int64_t image_width,
                                                           int64_t max_image_tiles, int64_t tile_size) {
    {
      auto possible_tile_arrangements = GetAllSupportedAspectRatios(max_image_tiles);
      std::vector<std::pair<int64_t, int64_t>> possible_canvas_sizes;

      for (const auto& arrangement : possible_tile_arrangements) {
        possible_canvas_sizes.emplace_back(arrangement.first * tile_size, arrangement.second * tile_size);
      }

      std::vector<double> scales;
      for (const auto& size : possible_canvas_sizes) {
        double scale_h = static_cast<double>(size.first) / image_height;
        double scale_w = static_cast<double>(size.second) / image_width;
        scales.push_back(std::min(scale_h, scale_w));
      }

      double selected_scale = 0;
      std::vector<double> upscaling_options;
      for (double scale : scales) {
        if (scale >= 1) {
          upscaling_options.push_back(scale);
        }
      }

      if (!upscaling_options.empty()) {
        selected_scale = *std::min_element(upscaling_options.begin(), upscaling_options.end());
      } else {
        std::vector<double> downscaling_options;
        for (double scale : scales) {
          if (scale < 1) {
            downscaling_options.push_back(scale);
          }
        }
        selected_scale = *std::max_element(downscaling_options.begin(), downscaling_options.end());
      }

      std::vector<std::pair<int64_t, int64_t>> chosen_canvas;
      for (size_t i = 0; i < scales.size(); ++i) {
        if (std::abs(scales[i] - selected_scale) < 1e-9) {
          chosen_canvas.push_back(possible_canvas_sizes[i]);
        }
      }

      if (chosen_canvas.size() > 1) {
        auto optimal_canvas = std::min_element(chosen_canvas.begin(), chosen_canvas.end(),
                                               [](const std::pair<int64_t, int64_t>& a, const std::pair<int64_t, int64_t>& b) {
                                                 return (a.first * a.second) < (b.first * b.second);
                                               });
        return *optimal_canvas;
      } else {
        return chosen_canvas[0];
      }
    }
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
                   ortc::Tensor<uint8_t>& padded_image) const {
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
                      std::pair<int64_t, int64_t>& aspect_ratio) const {
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
    std::unordered_map<std::string, ortx::AttrType> attrs = {{"height", new_height},
                                                             {"width", new_width},
                                                             {"interpolation", std::string("LINEAR")},
                                                             {"keep_aspect_ratio", int64_t(0)}};
    OrtxStatus status = resizer.Init(attrs);
    if (!status.IsOk()) {
      return status;
    }

    return resizer.Compute(image, resized_image);
  }

 public:
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

 private:
  int64_t max_image_tiles_{};
  std::pair<int64_t, int64_t> tile_size_{};
  std::string interpolation_{};

  Rescale rescale_;
  Normalize normalize_;
};

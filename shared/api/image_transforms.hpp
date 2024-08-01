// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "image_resample.h"

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
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "height") {
        height_ = std::get<int64_t>(value);
      } else if (key == "width") {
        width_ = std::get<int64_t>(value);
      } else if (key == "interpolation") {
        interpolation_ = std::get<std::string>(value);
        if (interpolation_ != "NEAREST" && interpolation_ != "LINEAR" && interpolation_ != "CUBIC") {
          return {kOrtxErrorInvalidArgument, "[Resize]: Invalid interpolation method"};
        }
      } else {
        return {kOrtxErrorInvalidArgument, "[Resize]: Invalid argument"};
      }
    }
    return {};
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

    int interp = IMAGING_TRANSFORM_NEAREST;
    if (interpolation_ == "NEAREST") {
      interp = IMAGING_TRANSFORM_NEAREST;
    } else if (interpolation_ == "LINEAR") {
      interp = IMAGING_TRANSFORM_BILINEAR;
    } else if (interpolation_ == "CUBIC") {
      interp = IMAGING_TRANSFORM_BICUBIC;
    } else if (interpolation_ == "LANCZOS") {
      interp = IMAGING_TRANSFORM_LANCZOS;
    } else {
      return {kOrtxErrorInvalidArgument, "[Resize]: Invalid interpolation method"};
    }

    float box[4] = {0.0f, 0.0f, static_cast<float>(width_), static_cast<float>(height_)};

    auto output_image = ImagingResample(rgb_image, static_cast<int>(width_), static_cast<int>(height_), interp, box);
    // cv::resize(image, output_image, {static_cast<int32_t>(width_), static_cast<int32_t>(height_)}, 0.0, 0.0, interp);
    ImagingDelete(rgb_image);

    auto* p_output_image = output.Allocate({height_, width_, c});
    for (auto i = height_ - height_; i < height_; ++i) {
      for (auto j = width_ - width_; j < width_; ++j) {
        auto c0_index = i * width_ * c + j * c;
        std::memcpy(p_output_image + c0_index, output_image->image[i] + j * 4, c);
      }
    }

    ImagingDelete(output_image);
    return {};
  }

 private:
  int64_t height_{256};
  int64_t width_{256};
  std::string interpolation_{"CUBIC"};  // LINEAR, NEAREST, CUBIC
};

struct Rescale {
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "scale") {
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
          p_output_image[c0_index + l] = input_data[c0_index + l] * scale_;
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
      } else {
        return {kOrtxErrorInvalidArgument, "[Normalize]: Invalid argument"};
      }
    }

    return {};
  }

  OrtxStatus Compute(const ortc::Tensor<float>& input, ortc::Tensor<float>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Normalize]: Only raw image formats"};
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
          p_output_image[c0_index + l] = (input_data[c0_index + l] - mean_[l]) / std_[l];
        }
      }
    }

    return {};
  }

 private:
  std::vector<float> mean_{0.48145466f, 0.4578275f, 0.40821073f};
  std::vector<float> std_{0.26862954f, 0.26130258f, 0.27577711f};
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

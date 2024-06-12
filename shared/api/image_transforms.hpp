// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

inline OrtxStatus convert_to_rgb(const ortc::Tensor<uint8_t>& input,
                                 ortc::Tensor<uint8_t>& output) {
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
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input,
                     ortc::Tensor<uint8_t>& output) {
    auto& dimensions = input.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Resize]: Only raw image formats"};
    }

    auto* input_data = input.Data();
    auto h = dimensions[0];
    auto w = dimensions[1];
    auto c = dimensions[2];

    cv::Mat image(h, w, CV_8UC3, const_cast<uint8_t*>(input_data));
    cv::Mat output_image;
    cv::InterpolationFlags interp{};
    if (interpolation_ == "NEAREST") {
      interp = cv::INTER_NEAREST;
    } else if (interpolation_ == "LINEAR") {
      interp = cv::INTER_LINEAR;
    } else if (interpolation_ == "CUBIC") {
      interp = cv::INTER_CUBIC;
    } else {
      return {kOrtxErrorInvalidArgument, "[Resize]: Invalid interpolation method"};
    }

    cv::resize(image, output_image,
               {static_cast<int32_t>(width_), static_cast<int32_t>(height_)}, 0.0, 0.0,
               interp);

    auto* p_output_image = output.Allocate({height_, width_, c});
    std::memcpy(p_output_image, output_image.data, height_ * width_ * c);

    return {};
  }

 private:
  int64_t height_{256};
  int64_t width_{256};
  std::string interpolation_{"CUBIC"};  // LINEAR, NEAREST, CUBIC
};

struct Rescale {
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input,
                     ortc::Tensor<float>& output) {
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
  OrtxStatus Compute(const ortc::Tensor<float>& input,
                     ortc::Tensor<float>& output) {
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
  std::vector<float> mean_;
  std::vector<float> std_;
};

struct CenterCrop {
  //   # T.CenterCrop(224),
  // width, height = self.target_size, self.target_size
  // img_h, img_w = img.shape[-2:]
  // s_h = torch.div((img_h - height), 2, rounding_mode='trunc')
  // s_w = torch.div((img_w - width), 2, rounding_mode='trunc')
  // x = img[:, :, s_h:s_h + height, s_w:s_w + width]

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& input,
                     ortc::Tensor<uint8_t>& output) {
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

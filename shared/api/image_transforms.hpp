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
    auto [height, width] = std::make_tuple(height_, width_);

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
//    DumpTensor(output);

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

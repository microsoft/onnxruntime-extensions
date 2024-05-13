// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "status.h"

constexpr int max_crops = 16;
constexpr int num_img_tokens = 144;
constexpr int image_resized_width = 336;
constexpr int image_resized_height = 336;

constexpr double OPENAI_CLIP_MEAN[] = {0.48145466, 0.4578275, 0.40821073};
constexpr double OPENAI_CLIP_STD[] = {0.26862954, 0.26130258, 0.27577711};

OrtxStatus convert_to_rgb(const ortc::Tensor<uint8_t>& input,
                          ortc::Tensor<uint8_t>& output) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 4ULL || dimensions[3] != 3) {
    return {kOrtxErrorInvalidArgument, "[ConvertToRGB]: input is not (N, H, W, C)"};
  }

  std::uint8_t* p_output_image = output.Allocate(dimensions);
  auto* input_data = input.Data();
  auto n = dimensions[0];
  auto h = dimensions[1];
  auto w = dimensions[2];
  auto c = dimensions[3];

  // convert BGR channel layouts to RGB
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j < h; ++j) {
      for (int64_t k = 0; k < w; ++k) {
        auto c0_index = i * h * w * c + j * w * c + k * c;
        std::tie(p_output_image[c0_index], p_output_image[c0_index + 1], p_output_image[c0_index + 2]) =
            std::make_tuple(input_data[c0_index + 2], input_data[c0_index + 1], input_data[c0_index]);
      }
    }
  }

  // for (int64_t i = 0; i < n; ++i) {
  //   memcpy(p_output_image + i * c * h * w + 2 * h * w, input_data + i * c * h * w, h * w);
  //   memcpy(p_output_image + i * c * h * w + 1 * h * w, input_data + i * c * h * w + 1 * h * w, h * w);
  //   memcpy(p_output_image + i * c * h * w, input_data + i * c * h * w + 2 * h * w, h * w);
  // }

  return {};
}

cv::Mat padding_336(const cv::Mat& image) {
  // def padding_336(b):
  //     width, height = b.size
  //     tar = int(np.ceil(height / 336) * 336)
  //     top_padding = int((tar - height)/2)
  //     bottom_padding = tar - height - top_padding
  //     left_padding = 0
  //     right_padding = 0
  //     b = torchvision.transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

  //     return b
  auto height = image.cols;
  auto tar = static_cast<int32_t>(std::ceil(height / image_resized_height) * image_resized_height);
  auto top_padding = (tar - height) / 2;
  auto bottom_padding = tar - height - top_padding;

  cv::_OutputArray output;
  cv::copyMakeBorder(image, output, top_padding, bottom_padding, 0, 0, cv::BORDER_CONSTANT, {255, 255, 255});
  return output.getMat();
}

cv::Mat hd_transform(const cv::Mat& image, int hd_num) {
  //     width, height = img.size
  auto [width, height] = std::make_tuple(image.rows, image.cols);

  //     ratio = width / height if width >= height else height / width
  float ratio = 1.0f * width;
  if (width >= height) {
    ratio /= height;
  } else {
    ratio = 1.0f * height / width;
  }

  //     scale = 1
  //     while scale * np.ceil(scale / ratio) <= hd_num:
  //         scale += 1
  //     scale -= 1
  int scale = 1;
  while (scale * std::ceil(scale / ratio) <= hd_num) {
    scale += 1;
  }
  scale -= 1;

  //     new_w = int(scale * 336)
  //     new_h = int(new_w / ratio)
  int64_t new_w = scale * 336;
  int64_t new_h = static_cast<int64_t>(new_w / ratio);

  //     if width < height:
  //         new_w, new_h = new_h, new_w
  if (width < height) {
    std::swap(new_w, new_h);
  }

  //     img = torchvision.transforms.functional.resize(img, [new_h, new_w])
  std::vector<int32_t> height_x_width{static_cast<int32_t>(new_h),   // H
                                      static_cast<int32_t>(new_w)};  // W

  cv::Mat output_image;
  cv::resize(image, output_image,
             {static_cast<int32_t>(new_w), static_cast<int32_t>(new_h)}, 0.0, 0.0,
             cv::INTER_LINEAR);
  //     img = padding_336(img)

  //     return img
  return output_image;
}

OrtxStatus phi3_hd_transform(ortc::Tensor<uint8_t>& input,
                             ortc::Tensor<float>& pixel_values,
                             ortc::Tensor<int64_t>& image_sizes,
                             ortc::Tensor<int64_t>& num_img_takens) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 3ULL) {
    return {kOrtxErrorInvalidArgument, "[ImageDecoder]: Only raw image formats"};
  }

  // Normalize the pixel value with mean and var
  auto input_data = input.Data();
  auto h = dimensions[0];
  auto w = dimensions[1];
  auto c = dimensions[2];

  // Allocate memory for output tensors
  auto rgb_image_ptr = std::make_unique<uint8_t>(h * w * c);
  auto p_pixel_values = rgb_image_ptr.get();
  for (int64_t j = 0; j < h; ++j) {
    for (int64_t k = 0; k < w; ++k) {
      auto c0_index = j * w * c + k * c;
      p_pixel_values[c0_index] = (input_data[c0_index] - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
      p_pixel_values[c0_index + 1] = (input_data[c0_index + 1] - OPENAI_CLIP_MEAN[1]) / OPENAI_CLIP_STD[1];
      p_pixel_values[c0_index + 2] = (input_data[c0_index + 2] - OPENAI_CLIP_MEAN[2]) / OPENAI_CLIP_STD[2];
    }
  }

  std::vector<int32_t> height_x_width{static_cast<int32_t>(h),   // H
                                      static_cast<int32_t>(w)};  // W

  cv::Mat rgb_image(height_x_width, CV_8UC3, p_pixel_values);
  // elems = [HD_transform(im, hd_num = self.num_crops) for im in images]
  auto elem = hd_transform(rgb_image, max_crops);
  // # tensor transform and normalize
  // hd_images = [img_processor(im) for im in elems]
  // # create global image
  // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336), mode='bicubic',).to(im.dtype) for im in hd_images]
  cv::Mat global_image;
  cv::resize(elem, global_image, {336, 336}, 0.0, 0.0, cv::INTER_CUBIC);

  int64_t shape[2];
  // # [(3, h, w)], where h, w is multiple of 336
  // shapes = [[im.size(1), im.size(2)] for im in hd_images]
  {
    auto shapes = image_sizes.Allocate({2});
    shapes[0] = input.Shape()[0];
    shapes[1] = input.Shape()[1];
  }
  // num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
  {
    auto n_tokens = num_img_takens.Allocate({1});
    auto [h_t, w_t] = std::make_tuple(image_sizes.Data()[0], image_sizes.Data()[1]);
    *n_tokens = int64_t((h_t / 336 * w_t / 336 + 1) * 144 + 1 + (h_t / 336 + 1) * 12);
  }
  // # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
  // # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
  // hd_images_reshape = [im.reshape(1, 3, h//336, 336, w//336, 336).permute(0,2,4,1,3,5).reshape(-1, 3, 336, 336).contiguous() for im, (h, w) in zip(hd_images, shapes)]
  // # concat global image and local image
  // hd_images_reshape = [torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
  // # pad to max_num_crops
  // image_transformed = [pad_to_max_num_crops_tensor(im, self.num_crops+1) for im in hd_images_reshape]
  // image_transformed = torch.stack(image_transformed, dim=0)
  // padded_images = image_transformed
  std::vector<int64_t> padded_image_shape = {max_crops, 3, image_resized_height, image_resized_width};
  auto* output_pixel = pixel_values.Allocate(padded_image_shape);
  // Copy the image pixel value from the global image
  const int image_c_size = 336 * 336 * 3;
  memcpy(output_pixel, global_image.data, image_c_size);
  auto num_crops = shape[0] / 336;
  std::uint8_t* image_transformed = nullptr;
  for (int i = 0; i < num_crops; ++i) {
    memcpy(output_pixel + (i + 1 * image_c_size), image_transformed + i * image_c_size, image_c_size);
  }

  memset(output_pixel + num_crops * image_c_size, 0, image_c_size * (max_crops - num_crops));

  // image_sizes = shapes
  return {};
}

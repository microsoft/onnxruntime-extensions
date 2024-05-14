// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "status.h"

constexpr int max_crops = 16;
constexpr int num_img_tokens = 144;
constexpr int image_resized_width = 336;
constexpr int image_resized_height = 336;

constexpr float OPENAI_CLIP_MEAN[] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float OPENAI_CLIP_STD[] = {0.26862954f, 0.26130258f, 0.27577711f};

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

inline cv::Mat padding_336(const cv::Mat& image) {
  // def padding_336(b):
  //     width, height = b.size
  //     tar = int(np.ceil(height / 336) * 336)
  //     top_padding = int((tar - height)/2)
  //     bottom_padding = tar - height - top_padding
  //     left_padding = 0
  //     right_padding = 0
  //     b = torchvision.transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

  //     return b
  float height = static_cast<float>(image.rows);
  int32_t tar = static_cast<int32_t>(std::ceil(height / image_resized_height) * image_resized_height);
  int32_t top_padding = static_cast<int32_t>((tar - height) / 2);
  int32_t bottom_padding = tar - image.rows - top_padding;

  cv::Mat output;
  cv::copyMakeBorder(image, output, top_padding, bottom_padding, 0, 0, cv::BORDER_CONSTANT, {255, 255, 255});
  return output;
}

inline cv::Mat hd_transform(const cv::Mat& image, int hd_num) {
  //     width, height = img.size
  auto [width, height] = std::make_tuple(image.cols, image.rows);

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
  int64_t new_w = scale * image_resized_width;
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
  return padding_336(output_image);
}

// Function to calculate 1D index from 3D indices
inline size_t Index3D(size_t i, size_t j, size_t k, size_t dim1, size_t dim2, size_t dim3) {
  return i * dim2 * dim3 + j * dim3 + k;
}

// Function to permute 3D array stored in 1D array from (X, Y, Z) to (Z, Y, X)
inline void Permute3DArray(const float* array, float* permutedArray, size_t X, size_t Y, size_t Z) {
  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {
      for (size_t z = 0; z < Z; ++z) {
        size_t oldIndex = Index3D(x, y, z, X, Y, Z);
        size_t newIndex = Index3D(z, y, x, Z, Y, X);
        permutedArray[newIndex] = array[oldIndex];
      }
    }
  }
}

inline OrtxStatus phi3_hd_transform(const ortc::Tensor<uint8_t>& input,
                             ortc::Tensor<float>& pixel_values,
                             ortc::Tensor<int64_t>& image_sizes,
                             ortc::Tensor<int64_t>& num_img_takens) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 3ULL) {
    return {kOrtxErrorInvalidArgument, "[hd_transform]: Only raw image formats"};
  }

  // Normalize the pixel value with mean and var
  auto input_data = input.Data();
  int32_t h = static_cast<int32_t>(dimensions[0]);
  int32_t w = static_cast<int32_t>(dimensions[1]);
  int32_t c = static_cast<int32_t>(dimensions[2]);
  std::vector<int32_t> height_x_width{static_cast<int32_t>(h),   // H
                                      static_cast<int32_t>(w)};  // W

  cv::Mat rgb_image(height_x_width, CV_8UC3, const_cast<uint8_t *>(input_data));
  // elems = [HD_transform(im, hd_num = self.num_crops) for im in images]
  auto elem = hd_transform(rgb_image, max_crops);
  // # tensor transform and normalize
  // hd_images = [img_processor(im) for im in elems]
  std::tie(w, h) = std::make_tuple(elem.cols, elem.rows);
  auto elem_image = elem.data;
  auto rgb_image_ptr = std::make_unique<float[]>(h * w * c);
  auto p_pixel_values = rgb_image_ptr.get();
  for (int64_t j = 0; j < h; ++j) {
    for (int64_t k = 0; k < w; ++k) {
      auto c0_index = j * w * c + k * c;
      p_pixel_values[c0_index] = (static_cast<float>(elem_image[c0_index]) / 255.f - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
      p_pixel_values[c0_index + 1] = (static_cast<float>(elem_image[c0_index + 1]) / 255.f - OPENAI_CLIP_MEAN[1]) / OPENAI_CLIP_STD[1];
      p_pixel_values[c0_index + 2] = (static_cast<float>(elem_image[c0_index + 2]) / 255.f - OPENAI_CLIP_MEAN[2]) / OPENAI_CLIP_STD[2];
    }
  }
  cv::Mat hd_image(h, w, CV_32FC3, p_pixel_values);
  // # create global image
  // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336), mode='bicubic',).to(im.dtype) for im in hd_images]
  cv::Mat global_image;
  cv::resize(hd_image, global_image, {image_resized_height, image_resized_width}, 0.0, 0.0, cv::INTER_CUBIC);

  int64_t shape[2];
  // # [(3, h, w)], where h, w is multiple of 336
  // shapes = [[im.size(1), im.size(2)] for im in hd_images]
  {
    auto shapes = image_sizes.Allocate({2});
    shapes[0] = shape[0] = hd_image.rows;
    shapes[1] = shape[1] = hd_image.cols;
  }
  // num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
  {
    auto n_tokens = num_img_takens.Allocate({1});
    auto [h_t, w_t] = std::make_tuple(image_sizes.Data()[0], image_sizes.Data()[1]);
    auto num_t = (static_cast<int32_t>(
      static_cast<int32_t>(h_t / image_resized_height) * w_t / image_resized_width) + 1) * 144 
      + 1 + static_cast<int32_t>(h_t / image_resized_height + 1) * 12;
    *n_tokens = static_cast<int64_t>(num_t);
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
  std::vector<int64_t> padded_image_shape = {max_crops + 1, 3, image_resized_height, image_resized_width};
  float* output_pixel = pixel_values.Allocate(padded_image_shape);
  // Copy the image pixel value from the global image
  const int image_c_size = image_resized_height * image_resized_width * 3;
  Permute3DArray(reinterpret_cast<float*>(global_image.data), output_pixel, image_resized_height, image_resized_width, 3);
  auto num_crops = static_cast<int>(std::ceil(float(shape[0]) / image_resized_height));
  float* image_transformed = reinterpret_cast<float*>(hd_image.data);
  for (int i = 0; i < num_crops; ++i) {
    Permute3DArray(image_transformed + i * image_c_size, output_pixel + (i + 1) * image_c_size, image_resized_height, image_resized_width, 3);
  }

  // padding the rest of the crops
  // pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
  memset(output_pixel + num_crops * image_c_size, 0, image_c_size * (max_crops - num_crops) * sizeof(float));

  // image_sizes = shapes
  return {};
}

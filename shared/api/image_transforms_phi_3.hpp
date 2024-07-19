// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "image_resample.h"

constexpr int max_crops = 16;
constexpr int num_img_tokens = 144;
constexpr int image_resized_width = 336;
constexpr int image_resized_height = 336;

constexpr float OPENAI_CLIP_MEAN[] = {0.48145466f, 0.4578275f, 0.40821073f};
constexpr float OPENAI_CLIP_STD[] = {0.26862954f, 0.26130258f, 0.27577711f};

inline Imaging padding_336_h(Imaging image) {
  // def padding_336(b):
  //     width, height = b.size
  //     tar = int(np.ceil(height / 336) * 336)
  //     top_padding = int((tar - height)/2)
  //     bottom_padding = tar - height - top_padding
  //     left_padding = 0
  //     right_padding = 0
  //     b = torchvision.transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding],
  //     fill=[255,255,255])

  //     return b
  float height = static_cast<float>(image->ysize);
  int32_t tar = static_cast<int32_t>(std::ceil(height / image_resized_height) * image_resized_height);
  if (tar == image->ysize) {
    return image;
  }
  int32_t top_padding = static_cast<int32_t>((tar - height) / 2);
  int32_t bottom_padding = tar - image->ysize - top_padding;

  Imaging output = ImagingNew("RGB", image->xsize, tar);
  for (int32_t i = 0; i < top_padding; ++i) {
    for (int32_t j = 0; j < image->xsize; ++j) {
      output->image[i][j * 4 + 0] = char(255);
      output->image[i][j * 4 + 1] = char(255);
      output->image[i][j * 4 + 2] = char(255);
      output->image[i][j * 4 + 3] = 0;  // unused
    }
  }
  for (int32_t i = top_padding; i < top_padding + image->ysize; ++i) {
    for (int32_t j = 0; j < image->xsize; ++j) {
      output->image[i][j * 4 + 0] = image->image[i - top_padding][j * 4];
      output->image[i][j * 4 + 1] = image->image[i - top_padding][j * 4 + 1];
      output->image[i][j * 4 + 2] = image->image[i - top_padding][j * 4 + 2];
      output->image[i][j * 4 + 3] = 0;  // unused
    }
  }
  for (int32_t i = top_padding + image->ysize; i < tar; ++i) {
    for (int32_t j = 0; j < image->xsize; ++j) {
      output->image[i][j * 4 + 0] = char(255);
      output->image[i][j * 4 + 1] = char(255);
      output->image[i][j * 4 + 2] = char(255);
      output->image[i][j * 4 + 3] = 0;  // unused
    }
  }

  ImagingDelete(image);
  return output;
}

inline Imaging padding_336_w(Imaging image) {
  float width = static_cast<float>(image->xsize);
  int32_t tar = static_cast<int32_t>(std::ceil(width / image_resized_width) * image_resized_width);
  if (tar == image->xsize) {
    return image;
  }

  int32_t left_padding = static_cast<int32_t>((tar - width) / 2);
  int32_t right_padding = tar - image->xsize - left_padding;

  Imaging output = ImagingNew("RGB", tar, image->ysize);
  for (int32_t i = 0; i < image->ysize; ++i) {
    for (int32_t j = 0; j < left_padding; ++j) {
      output->image[i][j * 4 + 0] = char(255);
      output->image[i][j * 4 + 1] = char(255);
      output->image[i][j * 4 + 2] = char(255);
      output->image[i][j * 4 + 3] = 0;  // unused
    }
    for (int32_t j = left_padding; j < left_padding + image->xsize; ++j) {
      output->image[i][j * 4 + 0] = image->image[i][(j - left_padding) * 4 + 0];
      output->image[i][j * 4 + 1] = image->image[i][(j - left_padding) * 4 + 1];
      output->image[i][j * 4 + 2] = image->image[i][(j - left_padding) * 4 + 2];
      output->image[i][j * 4 + 3] = 0;  // unused
    }
    for (int32_t j = left_padding + image->xsize; j < tar; ++j) {
      output->image[i][j * 4 + 0] = char(255);
      output->image[i][j * 4 + 1] = char(255);
      output->image[i][j * 4 + 2] = char(255);
      output->image[i][j * 4 + 3] = 0;  // unused
    }
  }

  ImagingDelete(image);
  return output;
}

inline Imaging hd_transform(Imaging image, int hd_num) {
  //     width, height = img.size
  auto [width, height] = std::make_tuple(image->xsize, image->ysize);

  //     ratio = width / height if width >= height else height / width
  double ratio = 1.0 * width;
  if (width >= height) {
    ratio /= height;
  } else {
    ratio = height / ratio;
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
  // std::vector<int32_t> height_x_width{static_cast<int32_t>(new_h),   // H
  //                                     static_cast<int32_t>(new_w)};  // W

  // cv::Mat output_image;
  // cv::resize(image, output_image, {static_cast<int32_t>(new_w), static_cast<int32_t>(new_h)}, 0.0, 0.0,
  //            cv::INTER_LINEAR);

  float box[4] = {0.0f, 0.0f, static_cast<float>(image->xsize), static_cast<float>(image->ysize)};
  auto output_image =
      ImagingResample(image, static_cast<int>(new_w), static_cast<int>(new_h), IMAGING_TRANSFORM_BILINEAR, box);
  ImagingDelete(image);

  //     img = padding_336(img)
  return width < height ? padding_336_w(output_image) : padding_336_h(output_image);
}

// Function to calculate 1D index from 3D indices
inline size_t Index3D(size_t i, size_t j, size_t k, size_t dim1, size_t dim2, size_t dim3) {
  return i * dim2 * dim3 + j * dim3 + k;
}

// Function to permute 3D array stored in 1D array from (X, Y, Z) to (Z, X, Y)
inline void Permute3DArray(const float* array, float* permutedArray, size_t X, size_t Y, size_t Z) {
  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {
      for (size_t z = 0; z < Z; ++z) {
        size_t oldIndex = Index3D(x, y, z, X, Y, Z);
        size_t newIndex = Index3D(z, x, y, Z, X, Y);
        permutedArray[newIndex] = array[oldIndex];
      }
    }
  }
}

inline OrtxStatus phi3_hd_transform(const ortc::Tensor<uint8_t>& input, ortc::Tensor<float>& pixel_values,
                                    ortc::Tensor<int64_t>& image_sizes, ortc::Tensor<int64_t>& num_img_tokens) {
  auto& dimensions = input.Shape();
  if (dimensions.size() != 3ULL) {
    return {kOrtxErrorInvalidArgument, "[hd_transform]: Only raw image formats"};
  }

  // Normalize the pixel value with mean and var
  auto input_data = input.Data();
  int32_t h = static_cast<int32_t>(dimensions[0]);
  int32_t w = static_cast<int32_t>(dimensions[1]);
  int32_t c = static_cast<int32_t>(dimensions[2]);
  // std::vector<int32_t> height_x_width{static_cast<int32_t>(h),   // H
  //                                     static_cast<int32_t>(w)};  // W

  // cv::Mat rgb_image(static_cast<int>(h), static_cast<int>(w), CV_8UC3, const_cast<uint8_t*>(input_data));
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

  // cv::Mat rgb_image(h, w, CV_8UC3, const_cast<uint8_t*>(input_data));
  // elems = [HD_transform(im, hd_num = self.num_crops) for im in images]
  auto elem = hd_transform(rgb_image, max_crops);
  // # tensor transform and normalize
  // hd_images = [img_processor(im) for im in elems]
  // std::tie(w, h) = std::make_tuple(elem->xsize, elem->ysize);
  // auto elem_image = elem->image;
  // auto rgb_image_ptr = std::make_unique<float[]>(h * w * c);
  // auto p_pixel_values = rgb_image_ptr.get();
  // for (int64_t j = 0; j < h; ++j) {
  //   for (int64_t k = 0; k < w; ++k) {
  //     auto c0_index = j * w * c + k * c;
  //     p_pixel_values[c0_index] =
  //         (static_cast<float>(elem_image[j][k * 4]) / 255.f - OPENAI_CLIP_MEAN[0]) / OPENAI_CLIP_STD[0];
  //     p_pixel_values[c0_index + 1] =
  //         (static_cast<float>(elem_image[j][k * 4 + 1]) / 255.f - OPENAI_CLIP_MEAN[1]) / OPENAI_CLIP_STD[1];
  //     p_pixel_values[c0_index + 2] =
  //         (static_cast<float>(elem_image[j][k * 4 + 2]) / 255.f - OPENAI_CLIP_MEAN[2]) / OPENAI_CLIP_STD[2];
  //   }
  // }

  std::tie(w, h) = std::make_tuple(elem->xsize, elem->ysize);
  uint8_t** elem_image = reinterpret_cast<uint8_t**>(elem->image);
  auto rgb_image_ptr = std::make_unique<float[]>(c * h * w);  // channel first
  auto p_pixel_values = rgb_image_ptr.get();
  for (int32_t k = 0; k < c; ++k) {
    for (int32_t i = 0; i < h; ++i) {
      for (int32_t j = 0; j < w; ++j) {
        p_pixel_values[k * h * w + i * w + j] =
            (static_cast<float>(elem_image[i][j * 4 + k]) / 255.f - OPENAI_CLIP_MEAN[k]) / OPENAI_CLIP_STD[k];
      }
    }
  }
  ImagingDelete(elem);

  auto shape = image_sizes.Allocate({2});
  {
    // # [(3, h, w)], where h, w is multiple of 336
    // shapes = [[im.size(1), im.size(2)] for im in hd_images]
    shape[0] = h;
    shape[1] = w;
  }

  // Debug code to check the image parity
  // auto rgb_image_ptr_debug = std::make_unique<float[]>(h * w * c);
  // Permute3DArray(p_pixel_values, rgb_image_ptr_debug.get(), h, w, c);

  auto image_size_1c = h * w;
  std::vector<Imaging> global_image(c);  // resample the image per channel
  for (int32_t k = 0; k < c; ++k) {
    // # create global image
    auto image_1c = ImagingNew("F", w, h);
    for (int32_t y = 0; y < h; ++y) {
      for (int32_t x = 0; x < w; ++x) {
        float* pixel = reinterpret_cast<float*>(image_1c->image[y]);
        *(pixel + x) = p_pixel_values[k * image_size_1c + y * w + x];
      }
    }
    // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336),
    //  mode='bicubic',).to(im.dtype) for im in hd_images]
    float box[]{0.0f, 0.0f, static_cast<float>(image_1c->xsize), static_cast<float>(image_1c->ysize)};
    global_image[k] =
        ImagingResample(image_1c, image_resized_width, image_resized_height, IMAGING_TRANSFORM_BICUBIC, box);
    ImagingDelete(image_1c);
  }

  // cv::Mat hd_image(h, w, CV_32FC3, p_pixel_values);
  // // # create global image
  // // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336),
  // // mode='bicubic',).to(im.dtype) for im in hd_images]
  // cv::Mat global_image;
  // cv::resize(hd_image, global_image, {image_resized_height, image_resized_width}, 0.0, 0.0, cv::INTER_CUBIC);
  {
    // num_img_tokens = [int((h//336*w//336+1)*144 + 1 + (h//336+1)*12) for h, w in shapes]
    auto n_tokens = num_img_tokens.Allocate({1});
    auto [h_t, w_t] = std::make_tuple(image_sizes.Data()[0], image_sizes.Data()[1]);
    auto num_t =
        (static_cast<int32_t>(static_cast<int32_t>(h_t / image_resized_height) * w_t / image_resized_width) + 1) * 144 +
        1 + static_cast<int32_t>(h_t / image_resized_height + 1) * 12;
    *n_tokens = static_cast<int64_t>(num_t);
  }
  // # reshape to channel dimension -> (num_images, num_crops, 3, 336, 336)
  // # (1, 3, h//336, 336, w//336, 336) -> (1, h//336, w//336, 3, 336, 336) -> (h//336*w//336, 3, 336, 336)
  // hd_images_reshape = [im.reshape(1, 3, h//336, 336, w//336, 336).permute(0,2,4,1,3,5).reshape(-1, 3, 336,
  // 336).contiguous() for im, (h, w) in zip(hd_images, shapes)] # concat global image and local image hd_images_reshape
  // = [torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)] # pad
  // to max_num_crops image_transformed = [pad_to_max_num_crops_tensor(im, self.num_crops+1) for im in
  // hd_images_reshape] image_transformed = torch.stack(image_transformed, dim=0) padded_images = image_transformed
  std::vector<int64_t> padded_image_shape = {max_crops + 1, 3, image_resized_height, image_resized_width};
  float* output_pixel = pixel_values.Allocate(padded_image_shape);
  // Copy the image pixel value from the global image
  // const int image_c_size = image_resized_height * image_resized_width * 3;
  // Permute3DArray(reinterpret_cast<float*>(global_image.data), output_pixel, image_resized_height,
  // image_resized_width,
  //                3);
  const int image_1c_size = image_resized_height * image_resized_width;
  for (auto i = c - c; i < c; ++i) {
    for (int y = 0; y < image_resized_height; ++y) {
      // memcpy(output_pixel + i * image_size_1c, image_transformed, image_size_1c * sizeof(float));
      auto image_transformed = reinterpret_cast<float*>(global_image[i]->image[y]);
      memcpy(output_pixel + i * image_1c_size + y * image_resized_width, image_transformed,
             image_resized_width * sizeof(float));
    }
  }

  for (auto img : global_image) {
    ImagingDelete(img);
  }

  auto num_crops = static_cast<int>((shape[0] / image_resized_height) * (shape[1] / image_resized_width));
  // float* image_transformed = reinterpret_cast<float*>(hd_image.data);
  // for (int i = 0; i < num_crops; ++i) {
  //   Permute3DArray(image_transformed + i * image_c_size, output_pixel + (i + 1) * image_c_size, image_resized_height,
  //   image_resized_width, 3);
  // }

  // chop the image into crops
  float* output_pixel_n_1 = output_pixel + image_1c_size * c;
  int m = static_cast<int>(shape[0] / image_resized_height);
  int n = static_cast<int>(shape[1] / image_resized_width);
  assert(m * n == num_crops);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int32_t k = 0; k < c; ++k) {
        // channel first
        int sub_index = (i * n + j) * image_1c_size * c + k * image_1c_size;
        for (int y = 0; y < image_resized_height; ++y) {
          for (int x = 0; x < image_resized_width; ++x) {
            output_pixel_n_1[sub_index + y * image_resized_width + x] =
                p_pixel_values[k * shape[0] * shape[1] + (i * image_resized_height + y) * shape[1] +
                               (j * image_resized_width + x)];
          }
        }
      }
    }
  }

  // for (int i = 0; i < m; ++i) {
  //   for (int j = 0; j < n; ++j) {
  //     int sub_index = (i * n + j) * image_c_size;
  //     for (int x = 0; x < image_resized_height; ++x) {
  //       for (int y = 0; y < image_resized_width; ++y) {
  //         for (int k = 0; k < 3; ++k) {  // Loop over channels
  //           output_pixel_n_1[sub_index + k * h * w + x * w + y] =
  //               image_transformed[((i * h + x) * shape[1] + (j * w + y)) * 3 + k];
  //         }
  //       }
  //     }
  //   }
  // }

  // padding the rest of the crops
  // pad = torch.zeros(max_crops - B, 3, H, W, dtype=images.dtype, device=images.device)
  memset(output_pixel_n_1 + num_crops * image_1c_size * c, 0,
         image_1c_size * c * (max_crops - num_crops) * sizeof(float));

  // image_sizes = shapes
  return {};
}

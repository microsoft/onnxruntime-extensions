// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <string>

#include "ext_status.h"
#include "op_def_struct.h"
#include "image_resample.h"

namespace ort_extensions {

class Phi4VisionDynamicPreprocess {

 public:
  Phi4VisionDynamicPreprocess() = default;

  /*
  def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
      best_ratio_diff = float('inf')
      best_ratio = (1, 1)
      area = width * height
      for ratio in target_ratios:
          target_aspect_ratio = ratio[0] / ratio[1]
          ratio_diff = abs(aspect_ratio - target_aspect_ratio)
          if ratio_diff < best_ratio_diff:
              best_ratio_diff = ratio_diff
              best_ratio = ratio
          elif ratio_diff == best_ratio_diff:
              if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                  best_ratio = ratio
      return best_ratio
  */
  std::pair<int, int> FindClosestAspectRatio(float aspect_ratio,
                                             const std::vector<std::pair<int, int>>& target_ratios,
                                             int width, int height, int image_size) const {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_ratio = {1, 1};
    int area = width * height;
    for (const auto& ratio : target_ratios) {
      float target_aspect_ratio = static_cast<float>(ratio.first) / ratio.second;
      float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
      if (ratio_diff < best_ratio_diff) {
        best_ratio_diff = ratio_diff;
        best_ratio = ratio;
      } else if (ratio_diff == best_ratio_diff) {
        if (area > 0.5 * image_size * image_size * ratio.first * ratio.second) {
          best_ratio = ratio;
        }
      }
    }
    return best_ratio;
  }

  OrtxStatus Compute(const ortc::Tensor<uint8_t>& ts_image,
                     ortc::Tensor<uint8_t>& resized_image,
                     ortc::Tensor<int64_t>& attention_mask) {

    auto& dimensions = ts_image.Shape();
    if (dimensions.size() != 3ULL) {
      return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: Only raw image formats"};
    }

    /*
    dyhd_base_resolution = 448

    # Dynamic HD
    base_resolution = dyhd_base_resolution
    images = [image.convert('RGB') for image in images]
    # cover 384 and 448 resolution
    mask_resolution = base_resolution // 14
    elems, image_attention_masks = [], []
    for im in images:
        elem, attention_mask = self.dynamic_preprocess(im, max_num=self.dynamic_hd, image_size=base_resolution, mask_size=mask_resolution)
        elems.append(elem)
        image_attention_masks.append(attention_mask)
    */
    const int32_t dyhd_base_resolution = 448;
    const int64_t mask_resolution = dyhd_base_resolution / 14;
    const uint8_t* input_data = ts_image.Data();
    int64_t h = ts_image.Shape()[0];
    int64_t w = ts_image.Shape()[1];
    int64_t c = ts_image.Shape()[2];
    Imaging image = ImagingNew("RGB", w, h);
    for (int64_t i = 0; i < h; ++i) {
      for (int64_t j = 0; j < w; ++j) {
        uint8_t* pixel = reinterpret_cast<uint8_t*>(image->image[i] + j * 4);
        pixel[0] = input_data[(i * w + j) * 3];
        pixel[1] = input_data[(i * w + j) * 3 + 1];
        pixel[2] = input_data[(i * w + j) * 3 + 2];
        pixel[3] = 0;  // unused
      }
    }

    Imaging elem{};
    std::vector<std::vector<int64_t>> image_attention_masks;
    OrtxStatus status = DynamicPreprocess(
      image, elem, image_attention_masks, 1, dynamic_hd_, dyhd_base_resolution, mask_resolution);
    if (!status.IsOk()) {
      return status;
    }

    auto* p_output_image = resized_image.Allocate({elem->ysize, elem->xsize, c});
    for (auto i = 0; i < elem->ysize; ++i) {
      for (auto j = 0; j < elem->xsize; ++j) {
        auto c0_index = i * elem->xsize * c + j * c;
        std::memcpy(p_output_image + c0_index, elem->image[i] + j * 4, c);
      }
    }

    ImagingDelete(image);
    ImagingDelete(elem);

    auto attention_mask_data = attention_mask.Allocate({
      static_cast<int64_t>(image_attention_masks.size()),
      static_cast<int64_t>(image_attention_masks[0].size())});

    for (size_t i = 0; i < image_attention_masks.size(); ++i) {
      for (size_t j = 0; j < image_attention_masks[i].size(); ++j) {
        attention_mask_data[i * image_attention_masks[i].size() + j] = image_attention_masks[i][j];
      }
    }

    return {};
  }

  /*
  def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=384, mask_size=27, use_thumbnail=True):
      orig_width, orig_height = image.size
      
      w_crop_num = math.ceil(orig_width/float(image_size))
      h_crop_num = math.ceil(orig_height/float(image_size))
      if w_crop_num * h_crop_num > max_num:

          aspect_ratio = orig_width / orig_height

          # calculate the existing image aspect ratio
          target_ratios = set(
              (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
              i * j <= max_num and i * j >= min_num)
          target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

          # find the closest aspect ratio to the target
          target_aspect_ratio = self.find_closest_aspect_ratio(
              aspect_ratio, target_ratios, orig_width, orig_height, image_size)

          # calculate the target width and height
          target_width = image_size * target_aspect_ratio[0]
          target_height = image_size * target_aspect_ratio[1]
          print(target_aspect_ratio)
      else:
          target_width = image_size * w_crop_num
          target_height = image_size * h_crop_num
          target_aspect_ratio = (w_crop_num, h_crop_num)

      # Calculate the ratio
      ratio_width = target_width / orig_width
      ratio_height = target_height / orig_height
      if ratio_width < ratio_height:
          new_size = (target_width, int(orig_height * ratio_width))
          padding_width = 0
          padding_height = target_height - int(orig_height * ratio_width)
      else:
          new_size = (int(orig_width * ratio_height), target_height)
          padding_width = target_width - int(orig_width * ratio_height)
          padding_height = 0

      attention_mask = torch.ones((int(mask_size*target_aspect_ratio[1]), int(mask_size*target_aspect_ratio[0])))
      if padding_width >= 14:
          attention_mask[:, -math.floor(padding_width/14):] = 0
      if padding_height >= 14:
          attention_mask[-math.floor(padding_height/14):,:] = 0
      assert attention_mask.sum() > 0

      if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
          raise ValueError(f'the aspect ratio is very extreme {new_size}') 

      image = torchvision.transforms.functional.resize(image, [new_size[1], new_size[0]],)

      resized_img = torchvision.transforms.functional.pad(image, [0, 0, padding_width, padding_height], fill=[255,255,255])

      return resized_img, attention_mask
    */

  OrtxStatus DynamicPreprocess(Imaging image,
    Imaging& resized_image, std::vector<std::vector<int64_t>>& attention_mask,
    int min_num=1, int max_num=12, int image_size=384, int mask_size=27 ) const {
    int orig_width = image->xsize;
    int orig_height = image->ysize;

    int w_crop_num = std::ceil(orig_width / static_cast<float>(image_size));
    int h_crop_num = std::ceil(orig_height / static_cast<float>(image_size));
    int target_width{}, target_height{};
    std::pair<int, int> target_aspect_ratio;
    if (w_crop_num * h_crop_num > max_num) {
      float aspect_ratio = static_cast<float>(orig_width) / orig_height;
      std::set<std::pair<int, int>> target_ratios;
      for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
          for (int j = 1; j <= n; ++j) {
            if (i * j <= max_num && i * j >= min_num) {
              target_ratios.insert({i, j});
            }
          }
        }
      }

      std::vector<std::pair<int, int>> target_ratios_sorted(target_ratios.begin(), target_ratios.end());
      std::sort(target_ratios_sorted.begin(), target_ratios_sorted.end(), [](const auto& a, const auto& b) {
        return a.first * a.second < b.first * b.second;
      });

      std::pair<int, int> target_aspect_ratio = FindClosestAspectRatio(aspect_ratio, target_ratios_sorted, orig_width, orig_height, image_size);
      target_width = image_size * target_aspect_ratio.first;
      target_height = image_size * target_aspect_ratio.second;
    } else {
      target_width = image_size * w_crop_num;
      target_height = image_size * h_crop_num;
      target_aspect_ratio = {w_crop_num, h_crop_num};
    }
    /*
        # Calculate the ratio
        ratio_width = target_width / orig_width
        ratio_height = target_height / orig_height
        if ratio_width < ratio_height:
            new_size = (target_width, int(orig_height * ratio_width))
            padding_width = 0
            padding_height = target_height - int(orig_height * ratio_width)
        else:
            new_size = (int(orig_width * ratio_height), target_height)
            padding_width = target_width - int(orig_width * ratio_height)
            padding_height = 0

    */
    float ratio_width = static_cast<float>(target_width) / orig_width;
    float ratio_height = static_cast<float>(target_height) / orig_height;
    std::pair<int, int> new_size;
    int padding_width, padding_height;
    if (ratio_width < ratio_height) {
      new_size = {target_width, static_cast<int>(orig_height * ratio_width)};
      padding_width = 0;
      padding_height = target_height - static_cast<int>(orig_height * ratio_width);
    } else {
      new_size = {static_cast<int>(orig_width * ratio_height), target_height};
      padding_width = target_width - static_cast<int>(orig_width * ratio_height);
      padding_height = 0;
    }

    /*
      attention_mask = torch.ones((int(mask_size*target_aspect_ratio[1]), int(mask_size*target_aspect_ratio[0])))
      if padding_width >= 14:
          attention_mask[:, -math.floor(padding_width/14):] = 0
      if padding_height >= 14:
          attention_mask[-math.floor(padding_height/14):,:] = 0
      assert attention_mask.sum() > 0

      if min(new_size[1], target_height) < 10 or min(new_size[0], target_width) < 10:
          raise ValueError(f'the aspect ratio is very extreme {new_size}') 
    */
    attention_mask.resize(mask_size * target_aspect_ratio.second, std::vector<int64_t>(mask_size * target_aspect_ratio.first, 1));
    if (padding_width >= 14) {
      for (int i = 0; i < mask_size * target_aspect_ratio.second; ++i) {
        for (int j = mask_size * target_aspect_ratio.first - std::floor(padding_width / 14); j < mask_size * target_aspect_ratio.first; ++j) {
          attention_mask[i][j] = 0;
        }
      }
    }
    if (padding_height >= 14) {
      for (int i = mask_size * target_aspect_ratio.second - std::floor(padding_height / 14); i < mask_size * target_aspect_ratio.second; ++i) {
        for (int j = 0; j < mask_size * target_aspect_ratio.first; ++j) {
          attention_mask[i][j] = 0;
        }
      }
    }
    assert(std::accumulate(attention_mask.begin(), attention_mask.end(), 0) > 0);

    if (std::min(new_size.second, target_height) < 10 || std::min(new_size.first, target_width) < 10) {
      return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: The aspect ratio is very extreme"};
    }

    // image = torchvision.transforms.functional.resize(image, [new_size[1], new_size[0]],)
    float box[4] = {0.0f, 0.0f, static_cast<float>(image->xsize), static_cast<float>(image->ysize)};
    auto output_image =
      ImagingResample(image, static_cast<int>(new_size.first), static_cast<int>(new_size.second), IMAGING_TRANSFORM_BILINEAR, box);

    // resized_img = torchvision.transforms.functional.pad(image, [0, 0, padding_width, padding_height], fill=[255,255,255])
    Imaging resized_img = ImagingNew("RGB", output_image->xsize + padding_width, output_image->ysize + padding_height);
    if (resized_img == nullptr) {
      return {kOrtxErrorOutOfMemory, "[Phi4VisionProcessor]: The aspect ratio is very extreme"};;
    }

    for (int i = 0; i < output_image->ysize; ++i) {
      for (int j = output_image->xsize; j < resized_img->xsize; ++j) {
        resized_img->image[i][j * 4 + 0] = char(255);
        resized_img->image[i][j * 4 + 1] = char(255);
        resized_img->image[i][j * 4 + 2] = char(255);
        resized_img->image[i][j * 4 + 3] = 0;  // unused
      }
    }

    for (int i = output_image->ysize; i < resized_img->ysize; ++i) {
      for (int j = 0; j < resized_img->xsize; ++j) {
        resized_img->image[i][j * 4 + 0] = char(255);
        resized_img->image[i][j * 4 + 1] = char(255);
        resized_img->image[i][j * 4 + 2] = char(255);
        resized_img->image[i][j * 4 + 3] = 0;  // unused
      }
    }

    ImagingDelete(output_image);
    resized_image = resized_img;
    return {};
  }

  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    return {};
  }

 private:
   int64_t dynamic_hd_{36};
};

class Phi4VisionProcessor {
 public:
  OrtxStatus Compute(const ortc::Tensor<float>& normalized_image,
                     const ortc::Tensor<int64_t>& image_attention_mask,
                     ortc::Tensor<float>& input_image_embeds,
                     ortc::Tensor<int64_t>& image_sizes,
                     ortc::Tensor<int64_t>& returned_image_attention_mask,
                     ortc::Tensor<int64_t>& num_img_tokens) const {
    const int64_t base_resolution = dyhd_base_resolution_;
    const int64_t mask_resolution = base_resolution / 14;
    /*
      hd_images = [img_processor(im) for im in elems]
      global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(base_resolution, base_resolution), mode='bicubic',).to(im.dtype) for im in hd_images]
      shapes = [[im.size(1), im.size(2)] for im in hd_images]
      mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
      global_attention_mask = [torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images]
    */
    auto normalized_image_data = normalized_image.Data();
    auto normalized_image_shape = normalized_image.Shape();
    auto [h, w, c] = std::make_tuple(normalized_image_shape[0], normalized_image_shape[1], normalized_image_shape[2]);
    // Imaging hd_image = ImagingNew("F", w, h);
    // for ()
    // for (int i = 0; i < hd_image->xsize; ++i) {
    //   for (int j = 0; j < hd_image->ysize; ++j) {
    //     float* pixel = reinterpret_cast<float*>(hd_image->image32[i]);
    //     *(pixel + j) = normalized_image_data[i * hd_image->xsize + j];
    //   }
    // }

    // float box[]{0.0f, 0.0f, static_cast<float>(hd_image->xsize), static_cast<float>(hd_image->ysize)};
    // Imaging global_image = ImagingResample(hd_image,
    //                                        base_resolution,
    //                                        base_resolution,
    //                                        IMAGING_TRANSFORM_BICUBIC,
    //                                        box);
    // ImagingDelete(hd_image);
    ortc::Tensor<float> ts_global_image{&CppAllocator::Instance()};
    auto global_image_data = ts_global_image.Allocate({c, base_resolution, base_resolution});
    auto image_size_1c = h * w;
    std::vector<Imaging> global_image(c);  // resample the image per channel
    for (int32_t k = 0; k < c; ++k) {
      // # create global image
      auto image_1c = ImagingNew("F", w, h);
      for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
          float* pixel = reinterpret_cast<float*>(image_1c->image[y]);
          *(pixel + x) = normalized_image_data[k * image_size_1c + y * w + x];
        }
      }
      // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336),
      //  mode='bicubic',).to(im.dtype) for im in hd_images]
      float box[]{0.0f, 0.0f, static_cast<float>(image_1c->xsize), static_cast<float>(image_1c->ysize)};
      global_image[k] =
          ImagingResample(image_1c, image_resized_width, image_resized_height, IMAGING_TRANSFORM_BICUBIC, box);
      if (global_image[k] == nullptr) {
        return {kOrtxErrorOutOfMemory, "[hd_transform]: Failed to allocate memory for global_image"};
      }
      ImagingDelete(image_1c);

      for (int i = 0; i < global_image[k]->ysize; ++i) {
        for (int j = 0; j < global_image[k]->xsize; ++j) {
          global_image_data[k * global_image[k]->ysize * global_image[k]->xsize + i * global_image[k]->xsize + j] =
            global_image[k]->image[i][j];
        }
      }

      ImagingDelete(global_image[k]);
    }

    /*
      hd_images_reshape = [im.reshape(1, 3,
                          h//base_resolution,
                          base_resolution,
                          w//base_resolution,
                          base_resolution
                          ).permute(0,2,4,1,3,5).reshape(-1, 3, base_resolution, base_resolution).contiguous() for im, (h, w) in zip(hd_images, shapes)]
      attention_masks_reshape = [mask.reshape(1,
                                h//mask_resolution,
                                mask_resolution,
                                w//mask_resolution,
                                mask_resolution
                                ).permute(0,1,3,2,4).reshape(-1, mask_resolution, mask_resolution).contiguous() for mask, (h, w) in zip(image_attention_masks, mask_shapes)]
      downsample_attention_masks = [mask[:,0::2,0::2].reshape(1,
                                    h//mask_resolution,
                                    w//mask_resolution,
                                    mask_resolution//2+mask_resolution%2,
                                    mask_resolution//2+mask_resolution%2
                                    ).permute(0,1,3,2,4) for mask, (h,w) in zip(attention_masks_reshape, mask_shapes)]
      downsample_attention_masks = [mask.reshape(mask.size(1)*mask.size(2), mask.size(3)*mask.size(4))for mask in downsample_attention_masks]
      num_img_tokens = [256 + 1 + int(mask.sum().item()) + int(mask[:,0].sum().item()) + 16 for mask in downsample_attention_masks]

      hd_images_reshape = [torch.cat([_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
      hd_masks_reshape = [torch.cat([_global_mask] + [_mask], dim=0) for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)]
      max_crops = max([img.size(0) for img in hd_images_reshape])
      image_transformed = [self.pad_to_max_num_crops(im, max_crops) for im in hd_images_reshape]
      image_transformed = torch.stack(image_transformed, dim=0)
      mask_transformed = [self.pad_mask_to_max_num_crops(mask, max_crops) for mask in hd_masks_reshape]
      mask_transformed = torch.stack(mask_transformed, dim=0)

      returned_input_image_embeds = image_transformed
      returned_image_sizes = torch.tensor(shapes, dtype=torch.long)
      returned_image_attention_mask = mask_transformed
      returned_num_img_tokens = num_img_tokens
    */
    ortc::Tensor<float> hd_images_reshape(&CppAllocator::Instance());
    SplitIntoTitles(normalized_image, hd_images_reshape, base_resolution, base_resolution);

    ortc::Tensor<int64_t> attention_masks_reshape(&CppAllocator::Instance());
    // SplitIntoTitles(image_attention_mask, attention_masks_reshape, mask_resolution, mask_resolution);
    // SplitIntoTitles only support 3d tensor, need to implement SplitIntoTitles for 2d tensor
    const int64_t tiles_w = w / mask_resolution;
    const int64_t tiles_h = h / mask_resolution;
    const int64_t crop_num = tiles_w * tiles_h;
    auto attention_mask_reshape_data = attention_masks_reshape.Allocate({crop_num, mask_resolution, mask_resolution});
    auto image_attention_mask_data = image_attention_mask.Data();
    auto image_attention_mask_shape = image_attention_mask.Shape();
    for (int i = 0; i < crop_num; ++i) {
      for (int j = 0; j < mask_resolution; ++j) {
        for (int k = 0; k < mask_resolution; ++k) {
          attention_mask_reshape_data[i * mask_resolution * mask_resolution + j * mask_resolution + k] =
            image_attention_mask_data[i * mask_resolution * mask_resolution + j * mask_resolution + k];
        }
      }
    }

    auto input_image_embeds_data = input_image_embeds.Allocate({
      hd_images_reshape.Shape()[0] + 1, 3, base_resolution, base_resolution});
    
    size_t global_image_size = ts_global_image.SizeInBytes();
    std::memcpy(input_image_embeds_data, ts_global_image.Data(), ts_global_image.SizeInBytes());
    std::memcpy(input_image_embeds_data + global_image_size, hd_images_reshape.Data(), hd_images_reshape.SizeInBytes());

    auto image_sizes_data = image_sizes.Allocate({hd_images_reshape.Shape()[0], 2});
    for (size_t i = 0; i < hd_images_reshape.Shape()[0]; ++i) {
      image_sizes_data[i * 2] = base_resolution;
      image_sizes_data[i * 2 + 1] = base_resolution;
    }

    std::vector<int64_t> global_attention_mask(mask_resolution * mask_resolution, 1);
    auto returned_image_attention_mask_data = returned_image_attention_mask.Allocate({
      attention_masks_reshape.Shape()[0] + 1, mask_resolution, mask_resolution});
    std::memcpy(returned_image_attention_mask_data,
                global_attention_mask.data(), global_attention_mask.size() * sizeof(int64_t));
    std::memcpy(returned_image_attention_mask_data + global_attention_mask.size(),
                attention_masks_reshape.Data(), attention_masks_reshape.SizeInBytes());

    auto num_img_tokens_data = num_img_tokens.Allocate({1});
    num_img_tokens_data[0] = 256 + 1 + std::accumulate(global_attention_mask.begin(), global_attention_mask.end(), 0) +
                             std::accumulate(global_attention_mask.begin(), global_attention_mask.end(), 0) + 16;

    return {};
  }

 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "dyhd_base_resolution") {
        dyhd_base_resolution_ = std::get<int64_t>(value);
      } else if (key == "interpolation") {
        interpolation_ = std::get<std::string>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: Invalid argument"};
      }
    }
    return {};
  }

  // Function to permute 3D array stored in 1D array from (X, Y, Z) to (Z, X, Y)
  static void Permute3DArray(const float* array, float* permutedArray, size_t X, size_t Y, size_t Z) {
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

 private:
  int64_t dyhd_base_resolution_{448};
  std::string interpolation_{"CUBIC"};
};

}  // namespace ort_extensions

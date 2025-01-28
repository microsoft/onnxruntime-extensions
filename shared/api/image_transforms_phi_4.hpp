// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <set>
#include <string>

#include "ext_status.h"
#include "op_def_struct.h"
#include "image_resample.h"
#include "narrow.h"

namespace ort_extensions {

class Phi4VisionDynamicPreprocess {
 public:
  Phi4VisionDynamicPreprocess() = default;
  OrtxStatus Compute(const ortc::Tensor<uint8_t>& ts_image, ortc::Tensor<uint8_t>& resized_image,
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
        elem, attention_mask = self.dynamic_preprocess(im, max_num=self.dynamic_hd, image_size=base_resolution,
    mask_size=mask_resolution) elems.append(elem) image_attention_masks.append(attention_mask)
    */
    const int64_t base_resolution = dyhd_base_resolution_;
    const int64_t mask_resolution = base_resolution / 14;
    const uint8_t* input_data = ts_image.Data();
    int64_t h = ts_image.Shape()[0];
    int64_t w = ts_image.Shape()[1];
    int64_t c = ts_image.Shape()[2];
    Imaging image = ImagingNew("RGB", ort_extensions::narrow<int>(w), ort_extensions::narrow<int>(h));
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
    OrtxStatus status =
      DynamicPreprocess(image, elem, image_attention_masks, 1, dynamic_hd_, base_resolution, mask_resolution);
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

    auto attention_mask_data = attention_mask.Allocate(
      {static_cast<int64_t>(image_attention_masks.size()), static_cast<int64_t>(image_attention_masks[0].size())});

    for (size_t i = 0; i < image_attention_masks.size(); ++i) {
      std::memcpy(
        attention_mask_data + i * image_attention_masks[i].size(),
        image_attention_masks[i].data(),
        image_attention_masks[i].size() * sizeof(image_attention_masks[i][0])
      );
    }

    return {};
  }

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
  std::pair<int64_t, int64_t> FindClosestAspectRatio(float aspect_ratio,
                                                     const std::vector<std::pair<int64_t, int64_t>>& target_ratios,
                                                     int64_t width, int64_t height, int64_t image_size) const {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int64_t, int64_t> best_ratio = {1, 1};
    int64_t area = width * height;
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

  OrtxStatus DynamicPreprocess(Imaging image, Imaging& resized_image, std::vector<std::vector<int64_t>>& attention_mask,
                               int64_t min_num = 1, int64_t max_num = 12, int64_t image_size = 384,
                               int64_t mask_size = 27) const {
    int64_t orig_width = image->xsize;
    int64_t orig_height = image->ysize;

    int64_t w_crop_num = static_cast<int64_t>(std::ceil(orig_width / static_cast<float>(image_size)));
    int64_t h_crop_num = static_cast<int64_t>(std::ceil(orig_height / static_cast<float>(image_size)));
    int64_t target_width{}, target_height{};
    std::pair<int64_t, int64_t> target_aspect_ratio;
    if (w_crop_num * h_crop_num > max_num) {
      float aspect_ratio = static_cast<float>(orig_width) / orig_height;
      std::set<std::pair<int64_t, int64_t>> target_ratios;
      for (int64_t n = min_num; n <= max_num; ++n) {
        for (int64_t i = 1; i <= n; ++i) {
          for (int64_t j = 1; j <= n; ++j) {
            if (i * j <= max_num && i * j >= min_num) {
              target_ratios.insert({i, j});
            }
          }
        }
      }

      std::vector<std::pair<int64_t, int64_t>> target_ratios_sorted(target_ratios.begin(), target_ratios.end());
      std::sort(target_ratios_sorted.begin(), target_ratios_sorted.end(),
          [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

      std::pair<int64_t, int64_t> target_aspect_ratio =
        FindClosestAspectRatio(aspect_ratio, target_ratios_sorted, orig_width, orig_height, image_size);
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
    std::pair<int64_t, int64_t> new_size;
    int64_t padding_width, padding_height;
    if (ratio_width < ratio_height) {
      new_size = {target_width, static_cast<int64_t>(orig_height * ratio_width)};
      padding_width = 0;
      padding_height = target_height - static_cast<int64_t>(orig_height * ratio_width);
    } else {
      new_size = {static_cast<int64_t>(orig_width * ratio_height), target_height};
      padding_width = target_width - static_cast<int64_t>(orig_width * ratio_height);
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
    attention_mask.resize(mask_size * target_aspect_ratio.second,
                          std::vector<int64_t>(mask_size * target_aspect_ratio.first, 1));
    if (padding_width >= 14) {
      for (int64_t i = 0; i < mask_size * target_aspect_ratio.second; ++i) {
        for (int64_t j = mask_size * target_aspect_ratio.first - static_cast<int64_t>(std::floor(padding_width / 14));
             j < mask_size * target_aspect_ratio.first; ++j) {
          attention_mask[i][j] = 0;
        }
      }
    }
    if (padding_height >= 14) {
      for (int64_t i = mask_size * target_aspect_ratio.second - static_cast<int64_t>(std::floor(padding_height / 14));
           i < mask_size * target_aspect_ratio.second; ++i) {
        for (int64_t j = 0; j < mask_size * target_aspect_ratio.first; ++j) {
          attention_mask[i][j] = 0;
        }
      }
    }

    if ((std::min)(new_size.second, target_height) < 10 || (std::min)(new_size.first, target_width) < 10) {
      return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: The aspect ratio is very extreme"};
    }

    // image = torchvision.transforms.functional.resize(image, [new_size[1], new_size[0]],)
    float box[4] = {0.0f, 0.0f, static_cast<float>(image->xsize), static_cast<float>(image->ysize)};
    auto output_image = ImagingResample(image, static_cast<int>(new_size.first), static_cast<int>(new_size.second),
                                        IMAGING_TRANSFORM_BILINEAR, box);

    /*
      resized_img = torchvision.transforms.functional.pad(
        image, [0, 0, padding_width, padding_height], fill=[255,255,255])
    */
    Imaging resized_img = ImagingNew(
      "RGB",
      output_image->xsize + ort_extensions::narrow<int>(padding_width),
      output_image->ysize + ort_extensions::narrow<int>(padding_height));
    if (resized_img == nullptr) {
      return {kOrtxErrorOutOfMemory, "[Phi4VisionProcessor]: The aspect ratio is very extreme"};
    }

    for (int i = 0; i < output_image->ysize; ++i) {
      for (int j = 0; j < output_image->xsize; ++j) {
        resized_img->image32[i][j] = output_image->image32[i][j];
      }
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
    for (const auto& [key, value] : attrs) {
      if (key == "dyhd_base_resolution") {
        dyhd_base_resolution_ = std::get<int64_t>(value);
      } else if (key == "dynamic_hd") {
        dynamic_hd_ = std::get<int64_t>(value);
      } else {
        return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: Invalid config: " + key};
      }
    }

    return {};
  }

 private:
  int64_t dynamic_hd_{36};
  int64_t dyhd_base_resolution_{448};
};

class Phi4VisionProcessor {
 public:
  OrtxStatus Compute(const ortc::Tensor<float>& hd_image,
                     const ortc::Tensor<int64_t>& image_attention_mask,
                     ortc::Tensor<float>& input_image_embeds,
                     ortc::Tensor<int64_t>& image_sizes,
                     ortc::Tensor<int64_t>& returned_image_attention_mask,
                     ortc::Tensor<int64_t>& num_img_tokens) const {
    const int32_t base_resolution = ort_extensions::narrow<int32_t>(dyhd_base_resolution_);
    const int64_t mask_resolution = base_resolution / 14;
    /*
      hd_images = [img_processor(im) for im in elems]
      global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(base_resolution, base_resolution),
      mode='bicubic',).to(im.dtype) for im in hd_images] shapes = [[im.size(1), im.size(2)] for im in hd_images]
      mask_shapes = [[mask.size(0), mask.size(1)] for mask in image_attention_masks]
      global_attention_mask = [torch.ones((1, mask_resolution, mask_resolution)) for _ in hd_images]
    */
    auto hd_image_data = hd_image.Data();
    auto hd_image_shape = hd_image.Shape();
    auto [h, w, c] = std::make_tuple(hd_image_shape[0], hd_image_shape[1], hd_image_shape[2]);
    ortc::Tensor<float> ts_global_image{&CppAllocator::Instance()};
    auto global_image_data = ts_global_image.Allocate({c, base_resolution, base_resolution});
    Imaging global_image_1c{};  // resample the image per channel
    for (int32_t k = 0; k < c; ++k) {
      // # create global image
      auto image_1c = ImagingNew("F", ort_extensions::narrow<int>(w), ort_extensions::narrow<int>(h));
      for (int32_t y = 0; y < h; ++y) {
        for (int32_t x = 0; x < w; ++x) {
          float* pixel = reinterpret_cast<float*>(image_1c->image32[y]);
          *(pixel + x) = hd_image_data[(y * w + x) * c + k];
        }
      }
      // global_image = [torch.nn.functional.interpolate(im.unsqueeze(0).float(), size=(336, 336),
      //  mode='bicubic',).to(im.dtype) for im in hd_images]
      float box[]{0.0f, 0.0f, static_cast<float>(image_1c->xsize), static_cast<float>(image_1c->ysize)};
      global_image_1c =
          ImagingResample(image_1c, base_resolution, base_resolution, IMAGING_TRANSFORM_BICUBIC, box);
      if (global_image_1c == nullptr) {
        return {kOrtxErrorOutOfMemory, "[hd_transform]: Failed to allocate memory for global_image"};
      }
      ImagingDelete(image_1c);

      auto global_image_size = global_image_1c->ysize * global_image_1c->xsize;
      for (int i = 0; i < global_image_1c->ysize; ++i) {
        for (int j = 0; j < global_image_1c->xsize; ++j) {
          global_image_data[k * global_image_size 
            + i * global_image_1c->xsize + j] = reinterpret_cast<float*>(global_image_1c->image32[i])[j];
        }
      }

      ImagingDelete(global_image_1c);
    }

    /*
      hd_images_reshape = [im.reshape(1, 3,
                                      h//base_resolution,
                                      base_resolution,
                                      w//base_resolution,
                                      base_resolution
                                      ).permute(0,2,4,1,3,5).reshape(
        -1, 3, base_resolution, base_resolution).contiguous() for im, (h, w) in zip(hd_images, shapes)]
      attention_masks_reshape = [mask.reshape(
        1,
        h//mask_resolution,
        mask_resolution,
        w//mask_resolution,
        mask_resolution
        ).permute(0,1,3,2,4).reshape(-1, mask_resolution, mask_resolution).contiguous(
          ) for mask, (h, w) in zip(image_attention_masks, mask_shapes)]

      hd_images_reshape = [torch.cat(
        [_global_image] + [_im], dim=0) for _global_image, _im in zip(global_image, hd_images_reshape)]
      hd_masks_reshape = [torch.cat(
        [_global_mask] + [_mask], dim=0) for _global_mask, _mask in zip(global_attention_mask, attention_masks_reshape)]
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
    SplitIntoTitles(hd_image, hd_images_reshape, base_resolution, base_resolution);

    ortc::Tensor<int64_t> attention_masks_reshape(&CppAllocator::Instance());
    // SplitIntoTitles only support 3d tensor, need to implement SplitIntoTitles for 2d tensor
    auto image_attention_mask_shape = image_attention_mask.Shape();
    std::tie(h, w) = std::make_tuple(image_attention_mask_shape[0], image_attention_mask_shape[1]);
    const int64_t tiles_w = w / mask_resolution;
    const int64_t tiles_h = h / mask_resolution;
    const int64_t crop_num = tiles_w * tiles_h;
    auto attention_mask_reshape_data = attention_masks_reshape.Allocate({crop_num, mask_resolution, mask_resolution});
    auto image_attention_mask_data = image_attention_mask.Data();
    const int64_t mask_size = mask_resolution * mask_resolution;
    for (int64_t i = 0; i < tiles_w; ++i) {
      for (int64_t j = 0; j < tiles_h; ++j) {
        for (int64_t x = 0; x < mask_resolution; ++x) {
          for (int64_t y = 0; y < mask_resolution; ++y) {
            attention_mask_reshape_data[(i * tiles_h + j)*mask_size + x * mask_resolution + y] =
                image_attention_mask_data[(j * mask_resolution + x) * w + i * mask_resolution + y];
          }
        }
      }
    }

    auto input_image_embeds_data =
      input_image_embeds.Allocate({hd_images_reshape.Shape()[0] + 1, 3, base_resolution, base_resolution});

    size_t global_image_size = ts_global_image.SizeInBytes();
    std::memcpy(input_image_embeds_data, ts_global_image.Data(), ts_global_image.SizeInBytes());
    std::memcpy(reinterpret_cast<char*>(input_image_embeds_data)
        + global_image_size, hd_images_reshape.Data(), hd_images_reshape.SizeInBytes());

    auto image_sizes_data = image_sizes.Allocate({2});
    image_sizes_data[0] = hd_image_shape[0];
    image_sizes_data[1] = hd_image_shape[1];

    std::vector<int64_t> global_attention_mask(mask_resolution * mask_resolution, 1);
    auto returned_image_attention_mask_data = returned_image_attention_mask.Allocate(
        {attention_masks_reshape.Shape()[0] + 1, mask_resolution, mask_resolution});
    std::memcpy(returned_image_attention_mask_data, global_attention_mask.data(),
                global_attention_mask.size() * sizeof(int64_t));
    std::memcpy(reinterpret_cast<char*>(returned_image_attention_mask_data)
                + global_attention_mask.size(), attention_masks_reshape.Data(), attention_masks_reshape.SizeInBytes());

    auto num_img_tokens_data = num_img_tokens.Allocate({1});
    num_img_tokens_data[0] = CountTokenNumber(image_attention_mask);

    return {};
  }
  /*
    (h, w) = mask_shapes()
    downsample_attention_mask = attention_masks_reshape[:,0::2,0::2].reshape(1,
                                      h//mask_resolution,
                                      w//mask_resolution,
                                      mask_resolution//2+mask_resolution%2,
                                      mask_resolution//2+mask_resolution%2
                                      ).permute(0,1,3,2,4)

    downsample_attention_masks = downsample_attention_mask.reshape(
      downsample_attention_mask.size(1)*downsample_attention_mask.size(2),
      downsample_attention_mask.size(3)*downsample_attention_mask.size(4))
    mask = downsample_attention_masks
    num_img_tokens = 256 + 1 + int(mask.sum().item()) + int(mask[:,0].sum().item()) + 16
  */
  int64_t CountTokenNumber(const ortc::Tensor<int64_t>& image_attention_mask) const {
    int64_t num_img_tokens = 0;
    auto attention_mask_shape = image_attention_mask.Shape();
    int64_t h = attention_mask_shape[0];
    int64_t w = attention_mask_shape[1];
    auto attention_mask_data = image_attention_mask.Data();
    // int(mask.sum().item()) 
    for (int64_t i = 0; i < h; ++i) {
      for (int64_t j = 0; j < w; ++j) {
        if (i % 2 == 0 && j % 2 == 0) {
          num_img_tokens += attention_mask_data[i * w + j];
        }
      }
    }

    // int(mask[:,0].sum().item())
    for (int64_t i = 0; i < h; i += 2) {
      num_img_tokens += attention_mask_data[i * w];
    }

    return 256 + 1 + num_img_tokens + 16;
  }

 public:
  template <typename DictT>
  OrtxStatus Init(const DictT& attrs) {
    for (const auto& [key, value] : attrs) {
      if (key == "dyhd_base_resolution") {
        dyhd_base_resolution_ = std::get<int64_t>(value);
      } else if (key == "interpolation") {
        std::string interpolation = std::get<std::string>(value);
        if (auto iter = Resize::InterpolationMethods().find(interpolation); iter != Resize::InterpolationMethods().end()) {
           interpolation_ = iter->second;
        } else {
          return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: Invalid interpolation: " + interpolation};
        }

      } else {
        return {kOrtxErrorInvalidArgument, "[Phi4VisionProcessor]: Invalid config: " + key};
      }
    }
    return {};
  }

 private:
  int64_t dyhd_base_resolution_{448};
  int interpolation_{IMAGING_TRANSFORM_BICUBIC};
};

}  // namespace ort_extensions

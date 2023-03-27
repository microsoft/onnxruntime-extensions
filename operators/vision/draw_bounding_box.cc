// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Note: make sure the input image has the format BGR

#include "draw_bounding_box.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <gsl/narrow>
#include <gsl/span>
#include <unordered_map>
#include <vector>

namespace ort_extensions {
namespace {
constexpr std::array<std::array<uint8_t, 3>, 10> KBGRColorMap = {{
    {{0, 0, 255}},      // Red
    {{0, 255, 0}},      // Green
    {{255, 0, 0}},      // Blue
    {{255, 255, 0}},    // Cyan
    {{255, 0, 255}},    // Magenta
    {{0, 0, 128}},      // Dark Red/Maroon
    {{0, 128, 0}},      // Dark Green/Lime
    {{128, 0, 0}},      // Dark Blue/Navy
    {{0, 0, 0}},        // Black
    {{128, 128, 128}},  // Gray
}};

// Used to store image data, width, height, and number of channels
struct ImageView {
  gsl::span<uint8_t> data;
  int64_t height;
  int64_t width;
  int64_t channels;
};

// To represent Boxes, [num of boxes, box_info]
class BoxArray {
 private:
  void SortBoxesByScore(const float* data) {
    boxes_by_score_.resize(NumBoxes());
    for (size_t i = 0; i < boxes_by_score_.size(); ++i) {
      boxes_by_score_[i] = gsl::make_span(data + i * ShapeAtDim(1), ShapeAtDim(1));
    }

    std::sort(boxes_by_score_.begin(), boxes_by_score_.end(),
              [](const gsl::span<const float>& first, const gsl::span<const float>& second) {
                return first[4] > second[4];
              });
  }

  [[nodiscard]] int64_t ShapeAtDim(int64_t dim) const {
    return shape_[dim];
  }

 public:
  BoxArray(const std::vector<int64_t>& shape, const float* data, BoundingBoxFormat bbox_mode) : shape_(shape), bbox_mode_(bbox_mode) {
    SortBoxesByScore(data);
  }

  [[nodiscard]] gsl::span<const float> GetBox(size_t index) const {
    return boxes_by_score_[index];
  }

  [[nodiscard]] int64_t NumBoxes() const {
    return shape_[0];
  }

  auto BBoxMode() const {
    return bbox_mode_;
  }

 private:
  const std::vector<int64_t>& shape_;
  std::vector<gsl::span<const float>> boxes_by_score_;
  BoundingBoxFormat bbox_mode_;
};

// Draw a line on the image
void DrawLineInHorizon(ImageView& image, int64_t x_start, int64_t y_start, int64_t line_length,
                       gsl::span<const uint8_t> color, int64_t thickness) {
  // boundary check
  thickness = std::clamp<int64_t>(thickness, 0, image.height - x_start);
  auto stride = image.width * image.channels;
  auto point_start = image.data.begin() + x_start * stride + y_start * image.channels;
  for (auto start = point_start; start < point_start + line_length * image.channels; start += image.channels) {
    std::copy_n(color.begin(), color.size(), start);
  }

  for (int64_t i = 1; i < thickness; ++i) {
    auto xy_point_start = point_start + i * stride;
    auto end = xy_point_start + line_length * image.channels;
    std::copy_n(point_start, line_length * image.channels, xy_point_start);
  }
}

void DrawLineInVertical(ImageView& image, int64_t x_start, int64_t y_start, int64_t line_length,
                        gsl::span<const uint8_t> color, int64_t thickness) {
  // boundary check
  thickness = std::clamp<int64_t>(thickness, 0, image.width - y_start);
  auto stride = image.width * image.channels;
  auto point_start = image.data.begin() + x_start * stride + y_start * image.channels;
  auto point_end = point_start + line_length * stride;

  for (int64_t i = 0; i < thickness; ++i) {
    std::copy_n(color.begin(), color.size(), point_start + i * image.channels);
  }
  for (auto xy_point_start = point_start + stride; xy_point_start < point_end; xy_point_start += stride) {
    std::copy_n(point_start, thickness * image.channels, xy_point_start);
  }
}

// Draw a box on the image with given thickness and color
void DrawBox(ImageView& image, gsl::span<const float> box, BoundingBoxFormat bbox_mode,
             gsl::span<const uint8_t> color, int64_t thickness) {
  // -------(1)------
  // |              |
  //(2)            (4)
  // |              |
  // -------(3)------
  float point_x1 = box[0];
  float point_x2 = box[1];
  float point_x3 = box[2];
  float point_x4 = box[3];
  if (bbox_mode == BoundingBoxFormat::Center_XYWH) {
    point_x1 = box[0] - point_x3 / 2;
    point_x2 = box[1] - point_x4 / 2;
    point_x3 = box[0] + point_x3 / 2;
    point_x4 = box[1] + point_x4 / 2;
  } else if (bbox_mode == BoundingBoxFormat::XYWH) {
    point_x1 = box[0];
    point_x2 = box[1];
    point_x3 = box[0] + box[2];
    point_x4 = box[1] + box[3];
  }
  int64_t x_start = static_cast<int64_t>(std::rintf(point_x1));
  int64_t y_start = static_cast<int64_t>(std::rintf(point_x2));
  int64_t x_end = static_cast<int64_t>(std::rintf(point_x3));
  int64_t y_end = static_cast<int64_t>(std::rintf(point_x4));

  auto offset = thickness / 2;
  x_start -= offset;
  y_start -= offset;
  x_end += (thickness - offset);
  y_end += (thickness - offset);

  x_start = std::max<int64_t>(x_start, 0L);
  y_start = std::max<int64_t>(y_start, 0L);
  x_end = std::min<int64_t>(x_end, image.height - 1);
  y_end = std::min<int64_t>(y_end, image.width - 1);

  thickness = std::clamp<int64_t>(thickness, 1, std::min(x_end - x_start, y_end - y_start));
  if (x_end - x_start < thickness || y_end - y_start < thickness) {
    // "invalid box, It's invalid to draw a point to represent a box.";
    return;
  }

  if (x_start + thickness >= image.height || y_start + thickness >= image.width) {
    // "invalid box, It's invalid to draw a point to represent a box.";
    return;
  }

  // line  (1) --
  DrawLineInHorizon(image, x_start, y_start, y_end - y_start, color, thickness);

  // line  (2) |--
  DrawLineInVertical(image, x_start, y_start, x_end - x_start, color, thickness);

  // line  (3) __
  DrawLineInHorizon(image, x_end - thickness, y_start, y_end - y_start, color, thickness);

  // line  (4) --|
  DrawLineInVertical(image, x_start, y_end - thickness, x_end - x_start, color, thickness);
}

void DrawBoxesForNumClasses(ImageView& image, const BoxArray& boxes, int64_t thickness) {
  std::unordered_map<float, size_t> color_used;

  for (size_t i = 0; i < boxes.NumBoxes(); ++i) {
    const auto box = boxes.GetBox(i);
    if (color_used.find(box[5]) == color_used.end()) {
      if (color_used.size() >= KBGRColorMap.size()) {
        // "The number of classes is larger than the number of colors in the color map.";
        continue;
      }
      color_used.emplace(box[5], color_used.size());
    }
    const auto color = KBGRColorMap[(static_cast<int64_t>(color_used[box[5]]))];
    DrawBox(image, box, boxes.BBoxMode(), color, thickness);
  }
}

void DrawBoxesByScore(ImageView& image, const BoxArray& boxes, int64_t thickness) {
  for (size_t i = 0; i < std::min<size_t>(KBGRColorMap.size(), boxes.NumBoxes()); ++i) {
    const auto color = KBGRColorMap[(static_cast<int64_t>(i))];
    DrawBox(image, boxes.GetBox(i), boxes.BBoxMode(), color, thickness);
  }
}

}  // namespace

void DrawBoundingBoxes::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_bgr = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtTensorDimensions dimensions_bgr(ort_, input_bgr);

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORTX_CXX_API_THROW("[DrawBoundingBoxes] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  const OrtValue* input_box = ort_.KernelContext_GetInput(context, 1ULL);
  const OrtTensorDimensions dimensions_box(ort_, input_box);
  // x,y, x/w y/h, score, class
  if (dimensions_box.size() != 2 || dimensions_box[1] != 6) {
    ORTX_CXX_API_THROW("[DrawBoundingBoxes] requires rank 2 input and the last dim should be 6.", ORT_INVALID_ARGUMENT);
  }
  BoxArray boxes(dimensions_box, ort_.GetTensorData<float>(input_box), bbox_mode_);
  int64_t image_size = dimensions_bgr[0] * dimensions_bgr[1] * dimensions_bgr[2];

  // Setup output & copy to destination
  // can we reuse the input buffer?
  const std::vector<int64_t> output_dims(dimensions_bgr);
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0,
                                                        output_dims.data(),
                                                        output_dims.size());

  auto* output_data = ort_.GetTensorMutableData<uint8_t>(output_value);
  const auto* input_data = ort_.GetTensorData<uint8_t>(input_bgr);
  // TODO: support batch in python side.
  // So we are hard-coding batch size to 1 for now.
  for (size_t batch_idx = 0; batch_idx < 1; ++batch_idx) {
    std::copy(input_data, input_data + image_size, output_data);
    auto data_span = gsl::make_span(output_data, image_size);
    ImageView image_view{data_span, dimensions_bgr[0], dimensions_bgr[1], dimensions_bgr[2]};
    if (colour_by_classes_) {
      DrawBoxesForNumClasses(image_view, boxes, gsl::narrow<int64_t>(thickness_));
    } else {
      DrawBoxesByScore(image_view, boxes, gsl::narrow<int64_t>(thickness_));
    }
    input_data += image_size;
    output_data += image_size;
  }
}

}  // namespace ort_extensions
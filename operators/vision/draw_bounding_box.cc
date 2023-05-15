// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Note: make sure the input image has the format BGR

#include "draw_bounding_box.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <gsl/narrow>
#include <gsl/span>
#include <gsl/span_ext>
#include <unordered_map>
#include <vector>
#include "exceptions.h"

namespace ort_extensions {
namespace {
constexpr std::array<std::array<uint8_t, 3>, 10> KBGRColorMap = {{
    {{0, 0, 255}},    // Red
    {{0, 255, 255}},  // Yellow
    {{0, 255, 0}},    // Lime
    {{255, 0, 0}},    // Blue
    {{255, 255, 0}},  // Cyan
    {{255, 0, 255}},  // Magenta
    {{0, 128, 255}},  // Orange
    {{0, 0, 128}},    // Maroon
    {{0, 128, 0}},    // Green
    {{128, 0, 0}},    // Navy
}};

// Used to store image data, width, height, and number of channels
struct ImageView {
  gsl::span<uint8_t> data;
  int64_t height;
  int64_t width;
  int64_t channels;
};

static constexpr size_t kBoxClassIndex = 5;
static constexpr size_t kBoxScoreIndex = 4;

// To represent Boxes, [num of boxes, box_info]
// box_info = [x1, y1, x2, y2, score, class]
class BoxArray {
 private:
  void SortBoxesByScore(gsl::span<const float> data) {
    boxes_by_score_.reserve(NumBoxes());
    for (size_t i = 0; i < NumBoxes(); ++i) {
      boxes_by_score_.push_back(data.subspan(i * shape_[1], shape_[1]));
    }

    std::sort(boxes_by_score_.begin(), boxes_by_score_.end(),
              [](const gsl::span<const float>& first, const gsl::span<const float>& second) {
                return first[kBoxScoreIndex] > second[kBoxScoreIndex];
              });
  }

 public:
  BoxArray(const std::vector<int64_t>& shape, gsl::span<const float> data, BoundingBoxFormat bbox_mode)
      : shape_(shape), bbox_mode_(bbox_mode) {
    SortBoxesByScore(data);
  }

  gsl::span<const float> GetBox(size_t index) const {
    assert(index < boxes_by_score_.size());
    return boxes_by_score_[index];
  }

  int64_t NumBoxes() const {
    return shape_[0];
  }

  BoundingBoxFormat BBoxMode() const {
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
  thickness = std::clamp<int64_t>(thickness, 0, image.height - y_start);
  auto stride = image.width * image.channels;
  auto point_start = image.data.begin() + y_start * stride + x_start * image.channels;
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
  thickness = std::clamp<int64_t>(thickness, 0, image.width - x_start);
  auto stride = image.width * image.channels;
  auto point_start = image.data.begin() + y_start * stride + x_start * image.channels;
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
  if (bbox_mode == BoundingBoxFormat::CENTER_XYWH) {
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

  // handle the case when the box is out of the image
  int64_t x_end = static_cast<int64_t>(std::clamp(std::round(point_x3), 0.F, static_cast<float>(image.width - 1)));
  int64_t y_end = static_cast<int64_t>(std::clamp(std::round(point_x4), 0.F, static_cast<float>(image.height - 1)));
  int64_t x_start = static_cast<int64_t>(std::clamp(std::round(point_x1), 0.F, static_cast<float>(x_end)));
  int64_t y_start = static_cast<int64_t>(std::clamp(std::round(point_x2), 0.F, static_cast<float>(y_end)));

  thickness = std::min<int64_t>(thickness, (std::min(x_end - x_start, y_end - y_start)));
  // skip invalid box, e.g. x_start >= image.width or y_start >= image.height
  // or x_end==x_start or y_end==y_start
  if (thickness < 1) {
    return;
  }
  
  // If not all filled
  if (thickness != (std::min(x_end - x_start, y_end - y_start))) {
    auto offset = thickness / 2;
    x_start -= offset;
    y_start -= offset;
    x_end += (thickness - offset);
    y_end += (thickness - offset);
  }

  // Clamp again to avoid out of bound with thickness
  x_end = std::clamp<int64_t>(x_end, 0, image.width - 1);
  y_end = std::clamp<int64_t>(y_end, 0, image.height - 1);
  x_start = std::clamp<int64_t>(x_start, 0, x_end);
  y_start = std::clamp<int64_t>(y_start, 0, y_end);

  auto box_width = x_end - x_start;
  auto box_height = y_end - y_start;
  // line  (1) --
  DrawLineInHorizon(image, x_start, y_start, box_width, color, thickness);

  // line  (2) |--
  DrawLineInVertical(image, x_start, y_start, box_height, color, thickness);

  // line  (3) __
  DrawLineInHorizon(image, x_start, y_end - thickness, box_width, color, thickness);

  // line  (4) --|
  DrawLineInVertical(image, x_end - thickness, y_start, box_height, color, thickness);
}

void DrawBoxesForNumClasses(ImageView& image, const BoxArray& boxes, int64_t thickness) {
  std::unordered_map<float, size_t> color_used;
  std::vector<std::pair<size_t, int64_t>> box_reverse;
  box_reverse.reserve(boxes.NumBoxes());
  for (size_t i = 0; i < boxes.NumBoxes(); ++i) {
    const auto box = boxes.GetBox(i);
    if (color_used.find(box[kBoxClassIndex]) == color_used.end()) {
      if (color_used.size() >= KBGRColorMap.size()) {
        // "The number of classes is larger than the number of colors in the color map.";
        continue;
      }
      color_used.emplace(box[kBoxClassIndex], color_used.size());
    }
    box_reverse.emplace_back(i, static_cast<int64_t>(color_used[box[kBoxClassIndex]]));
  }

  // A class which has higher score will be drawn on the top of the image.
  std::sort(box_reverse.begin(), box_reverse.end(),
            [](const std::pair<size_t, int64_t>& first_, const std::pair<size_t, int64_t>& second_) {
              return first_.second < second_.second;
            });
  for (int64_t i = static_cast<int64_t>(box_reverse.size()) - 1; i >= 0; --i) {
    auto [box_index, color_index] = box_reverse[i];
    const auto box = boxes.GetBox(box_index);
    const auto color = KBGRColorMap[color_index];
    DrawBox(image, box, boxes.BBoxMode(), color, thickness);
  }
}

void DrawBoxesByScore(ImageView& image, const BoxArray& boxes, int64_t thickness) {
  for (int64_t i = std::min<int64_t>(KBGRColorMap.size(), boxes.NumBoxes()) - 1; i >= 0; --i) {
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

  auto box_span = gsl::make_span(ort_.GetTensorData<float>(input_box), dimensions_box[0] * dimensions_box[1]);
  BoxArray boxes(dimensions_box, box_span, bbox_mode_);
  int64_t image_size = dimensions_bgr[0] * dimensions_bgr[1] * dimensions_bgr[2];

  // Setup output & copy to destination
  // can we reuse the input buffer?
  const std::vector<int64_t>& output_dims = dimensions_bgr;
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0,
                                                        output_dims.data(),
                                                        output_dims.size());

  auto* output_data = ort_.GetTensorMutableData<uint8_t>(output_value);
  const auto* input_data = ort_.GetTensorData<uint8_t>(input_bgr);

  std::copy(input_data, input_data + image_size, output_data);
  auto data_span = gsl::make_span(output_data, image_size);
  ImageView image_view{data_span, dimensions_bgr[0], dimensions_bgr[1], dimensions_bgr[2]};
  if (colour_by_classes_) {
    DrawBoxesForNumClasses(image_view, boxes, gsl::narrow<int64_t>(thickness_));
  } else {
    DrawBoxesByScore(image_view, boxes, gsl::narrow<int64_t>(thickness_));
  }
}

}  // namespace ort_extensions
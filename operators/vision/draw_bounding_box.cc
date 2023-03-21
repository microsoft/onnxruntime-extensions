// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Note make sure the input image has the format BGR

#include "draw_bounding_box.hpp"
#include <algorithm>
#include <gsl/narrow>
#include <gsl/span>
#include <vector>

namespace ort_extensions {
namespace {
constexpr uint8_t KColorMap[10][3] = {
    {255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 0}, {255, 0, 255}, {128, 0, 0}, {0, 128, 0}, {0, 0, 128}, {128, 128, 128}, {0, 0, 0}};

// Used to store image data, width, height, and number of channels
struct ImageView {
  uint8_t* data;
  int64_t height;
  int64_t width;
  int64_t channels;
};

// To represent Boxes, [num of boxes, box_info]
class RankTwoVector {
 public:
  RankTwoVector(const std::vector<int64_t>& shape, const float* data) : shape_(shape), data_(data) {
  }

  [[nodiscard]] const float* GetFirstDimDataPtr(size_t index) const {
    return data_ + index * shape_.back();
  }

  [[nodiscard]] int64_t ShapeAtDim(int64_t dim) const {
    return shape_[dim];
  }

 private:
  gsl::span<const int64_t> shape_;
  const float* data_;
};

// Draw a line on the image
void draw_line_horizon(ImageView& image, int64_t point_start, int64_t line_length, const uint8_t* color,
                       int64_t thickness) {
  auto stride = image.width * image.channels;
  for (int64_t i = 0; i < thickness; ++i) {
    auto* end = image.data + point_start + i * stride + line_length * image.channels;
    for (auto* start = image.data + point_start + i * stride; start < end; start += image.channels) {
      std::copy_n(color, image.channels, start);
    }
  }
}

void draw_line_vertical(ImageView& image, int64_t point_offset, int64_t line_length, const uint8_t* color,
                        int64_t thickness) {
  auto* point_start = image.data + point_offset;
  auto* point_end = point_start + line_length * image.width * image.channels;
  for (; point_start < point_end; point_start += image.width * image.channels) {
    for (int64_t i = 0; i < thickness; ++i) {
      // no boundary check
      std::copy_n(color, image.channels, point_start + i * image.channels);
    }
  }
}

// Draw a box on the image with given thickness and color
void draw_box(ImageView& image, const float* box,
              const uint8_t* color, int64_t thickness) {
  // -------(1)------
  // |              |
  //(2)            (4)
  // |              |
  // -------(3)------
  int64_t x_start = static_cast<int64_t>(box[0]);
  int64_t y_start = static_cast<int64_t>(box[1]);
  int64_t x_end = static_cast<int64_t>(box[2]);
  int64_t y_end = static_cast<int64_t>(box[3]);

  auto half_thickness = (thickness - 1) / 2;
  x_start -= half_thickness;
  y_start -= half_thickness;
  x_end += half_thickness;
  y_end += half_thickness;

  x_start = std::max(x_start, 0L);
  y_start = std::max(y_start, 0L);
  x_end = std::min(x_end, image.height - 1);
  y_end = std::min(y_end, image.width - 1);

  if (x_end - x_start < thickness || y_end - y_start < thickness) {
    // "invalid thickness, The box is too small to draw.";
    return;
  }

  if (x_start + thickness >= image.height || y_start + thickness >= image.width) {
    // "invalid thickness, The box is too small to draw.";
    return;
  }

  // BGR image, channel last
  int64_t x_start_offset = x_start * image.channels;
  int64_t x_end_offset = (x_end - thickness) * image.channels;
  int64_t y_start_offset = y_start * image.channels;
  int64_t y_end_offset = (y_end - thickness) * image.channels;

  // line  (1) --
  int64_t point_start = x_start_offset * image.width + y_start_offset;
  draw_line_horizon(image, point_start, y_end - y_start, color, thickness);

  // line  (2) |--
  draw_line_vertical(image, point_start, x_end - x_start, color, thickness);

  // line  (3) __
  point_start = x_end_offset * image.width + y_start_offset;
  draw_line_horizon(image, point_start, y_end - y_start, color, thickness);

  // line  (4) --|
  point_start = x_start_offset * image.width + y_end_offset;
  draw_line_vertical(image, point_start, x_end - x_start, color, thickness);
}

void draw_box_for_num_classes(ImageView& image, const RankTwoVector& boxes, int64_t thickness) {
  for (size_t i = 0; i < boxes.ShapeAtDim(0); ++i) {
    const auto* box = boxes.GetFirstDimDataPtr(i);
    const auto* color = KColorMap[(static_cast<int64_t>(box[5]) % 10)];
    draw_box(image, box, color, thickness);
  }
}

void draw_box_by_score(ImageView& image, const RankTwoVector& boxes, int64_t thickness) {
  std::vector<const float*> boxes_by_score(boxes.ShapeAtDim(0));
  for (size_t i = 0; i < boxes.ShapeAtDim(0); ++i) {
    boxes_by_score[i] = boxes.GetFirstDimDataPtr(i);
  }
  std::sort(boxes_by_score.begin(), boxes_by_score.end(), [](const float* a, const float* b) {
    return a[4] > b[4];
  });
  for (size_t i = 0; i < boxes_by_score.size(); ++i) {
    const auto* box = boxes_by_score[i];
    const auto* color = KColorMap[(static_cast<int64_t>(box[5]) % 10)];
    draw_box(image, box, color, thickness);
  }
}

}  // namespace

void DrawBoundingBox::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_bgr = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtTensorDimensions dimensions_bgr(ort_, input_bgr);

  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORTX_CXX_API_THROW("[DrawBoundingBox] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  const OrtValue* input_box = ort_.KernelContext_GetInput(context, 1ULL);
  const OrtTensorDimensions dimensions_box(ort_, input_box);
  // xmin, ymin, xmax, ymax, score, class
  if (dimensions_box.size() != 2 || dimensions_box[1] != 6) {
    ORTX_CXX_API_THROW("[DrawBoundingBox] requires rank 2 input and the last dim should be 6.", ORT_INVALID_ARGUMENT);
  }
  RankTwoVector boxes(dimensions_box, ort_.GetTensorData<float>(input_box));

  // Setup output & copy to destination
  // How can we reuse the input buffer?
  const std::vector<int64_t> output_dims(dimensions_bgr.begin(), dimensions_bgr.end());
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0,
                                                        output_dims.data(),
                                                        output_dims.size());

  auto* data = ort_.GetTensorMutableData<uint8_t>(output_value);
  const auto* input_data = ort_.GetTensorData<uint8_t>(input_bgr);
  std::copy(input_data, input_data + output_dims[0] * output_dims[1] * output_dims[2], data);
  int64_t image_size = output_dims[0] * output_dims[1] * output_dims[2];
  // TODO: support batch in python side.
  // So we are hard coding batch size to 1 for now.
  for (size_t batch_idx = 0; batch_idx < 1; ++batch_idx) {
    ImageView image_view{data + batch_idx * image_size, dimensions_bgr[0], dimensions_bgr[1], dimensions_bgr[2]};
    if (colour_by_classes_) {
      draw_box_for_num_classes(image_view, boxes, gsl::narrow<int64_t>(thickness_));
    } else {
      draw_box_by_score(image_view, boxes, gsl::narrow<int64_t>(thickness_));
    }
  }
}

}  // namespace ort_extensions
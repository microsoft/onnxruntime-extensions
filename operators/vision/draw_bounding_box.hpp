// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"

namespace ort_extensions {

// https://keras.io/api/keras_cv/bounding_box/formats/
// Regarding how the coordinates are stored in the bounding box array
enum class BoundingBoxFormat {
  XYWH,
  XYXY,
  CENTER_XYWH,
};

struct DrawBoundingBoxes : BaseKernel {
  DrawBoundingBoxes(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    thickness_ = TryToGetAttributeWithDefault<int64_t>("thickness", 4);
    num_classes_ = static_cast<int32_t>(TryToGetAttributeWithDefault<int64_t>("num_classes", 10));
    auto mode = TryToGetAttributeWithDefault<std::string>("mode", "XYXY");
    if (mode == "XYXY") {
      bbox_mode_ = BoundingBoxFormat::XYXY;
    } else if (mode == "XYWH") {
      bbox_mode_ = BoundingBoxFormat::XYWH;
    } else if (mode == "CENTER_XYWH") {
      bbox_mode_ = BoundingBoxFormat::CENTER_XYWH;
    } else {
      ORTX_CXX_API_THROW("[DrawBoundingBoxes] mode should be one of [XYXY, XYWH, CENTER_XYWH].", ORT_INVALID_ARGUMENT);
    }
    auto colour_by_classes = TryToGetAttributeWithDefault<int64_t>("colour_by_classes", 1);
    colour_by_classes_ = colour_by_classes > 0;
    if (thickness_ <= 0) {
      ORTX_CXX_API_THROW("[DrawBoundingBoxes] thickness of box should >= 1.", ORT_INVALID_ARGUMENT);
    }
  }

  void Compute(const ortc::Tensor<uint8_t>& input_bgr,
               const ortc::Tensor<float>& input_box,
               ortc::Tensor<uint8_t>& output) const;

 private:
  int64_t thickness_;
  int64_t num_classes_;
  bool colour_by_classes_;
  BoundingBoxFormat bbox_mode_;
};

}  // namespace ort_extensions

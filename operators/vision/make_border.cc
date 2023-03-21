// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Note make sure the input image has the format BGR

#include "make_border.hpp"
#include <algorithm>
#include <vector>

namespace ort_extensions {

void MakeBorder::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input_bgr = ort_.KernelContext_GetInput(context, 0ULL);
  const OrtTensorDimensions dimensions_bgr(ort_, input_bgr);
  if (dimensions_bgr.size() != 3 || dimensions_bgr[2] != 3) {
    // expect {H, W, C} as that's the inverse of what decode_image produces.
    // we have no way to check if it's BGR or RGB though
    ORTX_CXX_API_THROW("[MakeBorder] requires rank 3 BGR input in channels last format.", ORT_INVALID_ARGUMENT);
  }

  std::array<int64_t, 4> border_size{0, 0, 0, 0};
  const OrtValue* input_1 = ort_.KernelContext_GetInput(context, 1ULL);
  auto* input_data_1 = ort_.GetTensorData<int64_t>(input_1);
  if (mode_ == "output_shape") {
    if (input_data_1[0] < dimensions_bgr[0] || input_data_1[1] < dimensions_bgr[1]) {
      ORTX_CXX_API_THROW(
          "[MakeBorder] requires rank 2 input with shape [height, width] "
          "and height >= input height and width >= input width.",
          ORT_INVALID_ARGUMENT);
    }
    border_size[0] = (input_data_1[0] - dimensions_bgr[0]) / 2;
    border_size[2] = (input_data_1[1] - dimensions_bgr[1]) / 2;
    border_size[1] = input_data_1[0] - dimensions_bgr[0] - border_size[0];
    border_size[3] = input_data_1[1] - dimensions_bgr[1] - border_size[2];
  } else if (mode_ == "border_size") {
    const auto* input_data_1 = ort_.GetTensorData<int64_t>(input_1);
    std::copy(input_data_1, input_data_1 + 4, border_size.begin());
  } else {
    ORTX_CXX_API_THROW("[MakeBorder] requires mode to be either 'output_shape' or 'border_size'.", ORT_INVALID_ARGUMENT);
  }

  // Setup output & copy to destination
  const std::vector<int64_t> output_dims{border_size[0] + dimensions_bgr[0] + border_size[1],
                                         border_size[2] + dimensions_bgr[1] + border_size[3], 3};
  OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());

  auto* data = ort_.GetTensorMutableData<uint8_t>(output_value);
  std::fill_n(data, output_dims[0] * output_dims[1] * output_dims[2], fill_value_);

  //skip the first border_size[0] rows
  data = data + border_size[0] * output_dims[2] * output_dims[1];

  const auto* input_data = ort_.GetTensorData<uint8_t>(input_bgr);
  for(size_t i=border_size[0]; i<border_size[0]+dimensions_bgr[0]; ++i) {
    std::copy(input_data, input_data + dimensions_bgr[1] * dimensions_bgr[2], data + border_size[2] * 3);
    input_data = input_data + dimensions_bgr[1] * dimensions_bgr[2];
    data = data + output_dims[1] * output_dims[2];
  }
}

}  // namespace ort_extensions
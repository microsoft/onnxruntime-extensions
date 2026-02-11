// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "ocos.h"
#include "rotary_impl.cuh"
#include "ortx_common.h"

namespace contrib {

/**
 *  Y = Rotary(X) is equivalent to if side == LEFT:
 *
 *   N = X.shape[-1]
 *   Y = X.copy()
 *   Y[...,:N/2] = X[...,N/2:]
 *   Y[...,N/2:] = -X[...,:N/2]
 *
 *   And the opposite if side == RIGHT:
 *
 *   N = X.shape[-1]
 *   Y = X.copy()
 *   Y[...,:N/2] = -X[...,N/2:]
 *   Y[...,N/2:] = X[...,:N/2]
 */
template <typename T>
struct Rotary {
  template <typename TDict>
  OrtxStatus OnModelAttach(const TDict& dict) {
    std::string empty;
    std::string side = dict.TryToGetAttributeWithDefault("side", empty);
    if (side == "left") {
      side_ = RotarySide::LEFT;
    } else if (side == "right") {
      side_ = RotarySide::RIGHT;
    } else {
      return {kOrtxErrorInvalidArgument, "side must be 'left' or 'right'."};
    }

    return {};
  }

  OrtxStatus Compute(Ort::Custom::CUDAKernelContext* ctx, const ortc::Tensor<T>& input,
                     const ortc::Tensor<int64_t>& split, ortc::Tensor<T>& output) const {
    const T* input_data = input.Data();
    auto input_shape = input.Shape();
    T* output_data = output.Allocate(input_shape);
    auto input_length = input.NumberOfElement();
    if (0 == input_length) {
      return {};
    }

    auto shape_split = split.Shape();
    if (shape_split.size() != 1 || shape_split[0] != 2) {
      return {kOrtxErrorInvalidArgument, "Rotary only works when there are two sides."};
    }
    const int64_t* split_data = split.Data();
    if (split_data[0] != split_data[1]) {
      return {kOrtxErrorInvalidArgument, "Only equal split is allowed."};
    }
    if (split_data[0] != split_data[1] != input_shape[input_shape.size() - 1]) {
      return {kOrtxErrorInvalidArgument, "Sum of the splits are not equal to the last dimension."};
    }

    LaunchRotaryKernel<T>(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), input_length,
                          static_cast<int>(input_shape[input_shape.size() - 1]), input_data, split_data, output_data,
                          side_);
    return {};
  }

  static OrtMemType GetInputMemoryType(size_t input_index) {
    if (input_index == 1)  // split
      return OrtMemType::OrtMemTypeCPUInput;
    return OrtMemType::OrtMemTypeDefault;
  }

 private:
  RotarySide side_;
};

}  // namespace contrib
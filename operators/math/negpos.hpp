// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

void neg_pos(const ortc::Tensor<float>& input,
             ortc::Tensor<float>& out0_tensor,
             ortc::Tensor<float>& out1_tensor) {
  int64_t size = input.NumberOfElement();
  float* out0 = out0_tensor.Allocate(input.Shape());
  float* out1 = out1_tensor.Allocate(input.Shape());
  const float* X = input.Data();
  // Do computation
  for (int64_t i = 0; i < size; i++) {
    if (X[i] > 0) {
      out0[i] = 0;
      out1[i] = X[i];
    } else {
      out0[i] = X[i];
      out1[i] = 0;
    }
  }
}

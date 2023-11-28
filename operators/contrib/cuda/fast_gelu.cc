// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fast_gelu.h"
#include <cuda.h>
#include <cuda_runtime.h>

namespace contrib {

/*
template <typename T>
OrtStatusPtr FastGelu<T>::Compute(const ortc::Tensor<T>& input,
                                  std::optional<const ortc::Tensor<T>*> bias,
                                  ortc::Tensor<T>& output) const {
  const T* input_data = input.Data();
  // const T* bias_data = bias.Data();
  T* output_data = output.Allocate(input.Shape());
  auto input_length = input.NumberOfElement();
  if (0 == input_length) {
    return nullptr;
  }
  const T* bias_data = bias.has_value()?(*bias)->Data():nullptr;
  auto bias_length = bias.has_value()?(*bias)->NumberOfElement():0;
  return nullptr;
}*/

} // contrib
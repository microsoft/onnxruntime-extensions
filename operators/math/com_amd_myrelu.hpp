// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <torch/torch.h>

#include "ocos.h"

// TODO: Good example for CPU/CUDA op
// https://github.com/microsoft/onnxruntime-extensions/pull/739/files

// TODO: Add DLPack support to ONNXRuntime-extensions for perf improvement
// https://github.com/microsoft/onnxruntime/pull/6968

// TODO: Make templates for Tensor<T>? Testing for Tensor<float>
// template <typename T>
OrtStatusPtr com_amd_myrelu(const ortc::Tensor<float>& input_ort,
                            ortc::Tensor<float>& out_ort) {

  int64_t input_size = input_ort.NumberOfElement();
  if (0 == input_size) {
    return nullptr;
  }

  // Massaging the input to Pytorch format
  torch::Tensor X = torch::empty(input_ort.Shape()).contiguous();
  memcpy(X.data_ptr<float>(), input_ort.Data(), input_size * sizeof(float)); // TODO: replace with todlpack + torch::Tensor

  // Do computation
  float* out_ort_ptr = out_ort.Allocate(input_ort.Shape());

  // Massaging the output to ORT format
  auto out_torch = torch::relu(X);
  memcpy(out_ort_ptr, out_torch.data_ptr<float>(), input_size * sizeof(float)); // TODO: replace with todlpack + ortc::Tensor conversion

  return nullptr;
}

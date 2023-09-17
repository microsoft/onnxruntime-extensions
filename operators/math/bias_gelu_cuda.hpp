// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"

void bias_gelu_cuda(const Ort::Custom::CudaContext& cuda_ctx,
                    const ortc::Tensor<float>& X,
                    const ortc::Tensor<float>& B,
                    ortc::Tensor<float>& Y);

void bias_gelu_cuda_shape_infer(const Ort::Custom::TensorShapeVec& input_shapes, Ort::Custom::TensorShapeVec& output_shape);
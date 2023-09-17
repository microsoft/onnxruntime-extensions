// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bias_gelu_cuda.hpp"
#include "bias_gelu_cuda_impl.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>

void bias_gelu_cuda(const Ort::Custom::CudaContext& cuda_ctx,
                    const ortc::Tensor<float>& X,
                    const ortc::Tensor<float>& B,
                    ortc::Tensor<float>& Y) {
    auto x_shape = X.Shape();
    launch_cuda_kernel(cuda_ctx.cuda_stream, X.NumberOfElement(), B.NumberOfElement(), X.Data(), B.Data(), Y.Allocate(x_shape));
}

void bias_gelu_cuda_shape_infer(const Ort::Custom::TensorShapeVec& input_shapes,
                                Ort::Custom::TensorShapeVec& output_shape) {
    auto num_input_shapes = input_shapes.size();
    auto num_output_shapes = output_shape.size();
    if (num_input_shapes > 0 && num_output_shapes > 0) {
      auto x_shape = input_shapes[0].GetShape();
      output_shape[0].SetShape(x_shape);
      /*std::vector<int64_t> y_shape{static_cast<int64_t>(input_shapes[0].GetElementCount())};
      output_shape[0].SetShape(y_shape);*/
      std::cout << "set x shape to y" << std::endl;
    }
}
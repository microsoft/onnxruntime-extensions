// #include "negpos.cuh"
#include "narrow.h"
#include "negpos_def.h"
#include <cuda.h>
#include <cuda_runtime.h>

void neg_pos_impl(cudaStream_t stream,
                  const float* input, float* pos_out, float* neg_out, int size);

OrtStatusPtr neg_pos_cuda(const Ort::Custom::CudaContext& ctx,
                          const ortc::Tensor<float>& input,
                          ortc::Tensor<float>& out0_tensor,
                          ortc::Tensor<float>& out1_tensor) {
  auto size = ort_extensions::narrow<int>(input.NumberOfElement());
  float* out0 = out0_tensor.Allocate(input.Shape());
  float* out1 = out1_tensor.Allocate(input.Shape());
  const float* X = input.Data();

  neg_pos_impl(reinterpret_cast<cudaStream_t>(ctx.cuda_stream), X, out0, out1, size);
  return nullptr;
}

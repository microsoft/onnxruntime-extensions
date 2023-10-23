#include "negpos.cuh"
#include "narrow.h"
#include "negpos.h"

OrtStatusPtr neg_pos_cuda(const Ort::Custom::CudaContext& ctx,
                          const ortc::Tensor<float>& input,
                          ortc::Tensor<float>& out0_tensor,
                          ortc::Tensor<float>& out1_tensor) {
  auto size = ort_extensions::narrow<int>(input.NumberOfElement());
  float* out0 = out0_tensor.Allocate(input.Shape());
  float* out1 = out1_tensor.Allocate(input.Shape());
  const float* X = input.Data();

  neg_pos_impl(ctx.cuda_stream, X, out0, out1, size);
  return nullptr;
}

#include "com_amd_myrelu.cuh"
#include "narrow.h"
#include "com_amd_myrelu_def.h"
#include <cuda.h>
#include <cuda_runtime.h>

OrtStatusPtr com_amd_myrelu_cuda(Ort::Custom::CUDAKernelContext* ctx,
                                 const ortc::Tensor<float>& input,
                                 ortc::Tensor<float>& out0_tensor) {
  // TODO: Properly implement CUDA version
  int64_t input_size = input_ort.NumberOfElement() * sizeof(float);
  if (0 == input_size) {
    return nullptr;
  }

  // Massaging the input to Pytorch format
  torch::Tensor X = torch::empty(input_ort.Shape()).contiguous();
  memcpy(X.data_ptr<float>(), input_ort.Data(), input_size); // TODO: replace with todlpack + torch::Tensor

  // Do computation
  float* out_ort_ptr = out_ort.Allocate(input_ort.Shape());

  com_amd_myrelu_impl(reinterpret_cast<cudaStream_t>(ctx->GetCudaStream()), X, out_ort_ptr, input_size);
  return nullptr;
}
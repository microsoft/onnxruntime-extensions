// #include <torch/extension.h>
#include <torch/torch.h>
#include <cuda_runtime.h>
#include "com_amd_myrelu.cuh"

__global__ void com_amd_myrelu_kernel(const float* input, float* out, int input_size) {
  // TODO: Properly implement CUDA version

  // Massaging the output to ORT format
  auto out_torch = torch::relu(input);
  memcpy(out, out_torch.data_ptr<float>(), input_size); // TODO: replace with todlpack + ortc::Tensor conversion
}

void com_amd_myrelu_impl(cudaStream_t stream,
                         const float* input, float* out, int size) {
  com_amd_myrelu_kernel<<<1, 1, 0, stream>>>(input, out, size);
}

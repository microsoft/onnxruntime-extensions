#include <cuda_runtime.h>
#include "negpos.cuh"

__global__ void neg_pos_kernel(const float* input, float* pos_out, float* neg_out, int size) {
  for (int i = 0; i < size; i++) {
    if (input[i] > 0) {
      pos_out[i] = input[i];
      neg_out[i] = 0;
    } else {
      pos_out[i] = 0;
      neg_out[i] = input[i];
    }
  }
}

void neg_pos_impl(cudaStream_t stream,
                  const float* input, float* pos_out, float* neg_out, int size) {
  neg_pos_kernel<<<1, 1, 0, stream>>>(input, pos_out, neg_out, size);
}

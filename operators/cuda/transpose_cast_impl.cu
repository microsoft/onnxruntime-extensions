// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "device_prop.cuh"
#include "utils.cuh"
#include "transpose_cast_impl.cuh"
#include "cuda_type.h"

using namespace Ort::Custom;

template <typename TOUT, typename TIN>
__global__ void TransposeCast2DKernel(TOUT *output_data, const TIN *input_data, int n_rows, int n_cols) {
  __shared__ TIN tile[TILE_DIM][TILE_DIM + 1];

  int x = blockIdx.x * TILE_DIM + threadIdx.x;
  int y = blockIdx.y * TILE_DIM + threadIdx.y;
  // int width = gridDim.x * TILE_DIM;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    tile[threadIdx.y + j][threadIdx.x] = input_data[(y + j) * n_cols + x];

  __syncthreads();

  x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose block offset
  y = blockIdx.x * TILE_DIM + threadIdx.y;

  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    output_data[(y + j) * n_rows + x] = (TOUT)(tile[threadIdx.x][threadIdx.y + j]);
}

template <typename TIN, typename TOUT>
cudaError_t _LaunchTransposeCast2DKernel(cudaStream_t stream, size_t n_rows, size_t n_cols, const TIN* input, TOUT* output) {
  constexpr int blockSize = 256;
  const int gridSize = (input_length + blockSize - 1) / blockSize;
  using TTIN = typename contrib::CudaT<TIN>::MappedType;
  using TTOUT = typename contrib::CudaT<TOUT>::MappedType;
  TransposeCast2DKernel<TTOUT, TTIN><<<gridSize, blockSize, 0, stream>>>(
    reinterpret_cast<TTOUT*>(output), reinterpret_cast<const TTIN*>(input),
    static_cast<int>(n_rows), static_cast<int>(n_cols));
  return cudaGetLastError();
}

template <>
cudaError_t LaunchTransposeCast2DKernel<float, ortc::MFloat16>(cudaStream_t stream, size_t n_rows, size_t n_cols, const float* input,  ortc::MFloat16* output) {
  return _LaunchTransposeCast2DKernel(stream, n_rows, n_cols, , input, output);
}

template <>
cudaError_t LaunchTransposeCast2DKernel<ortc::MFloat16, float>(cudaStream_t stream, size_t n_rows, size_t n_cols, const ortc::MFloat16* input, float* output) {
  return _LaunchTransposeCast2DKernel(stream, n_rows, n_cols, input, output);
}

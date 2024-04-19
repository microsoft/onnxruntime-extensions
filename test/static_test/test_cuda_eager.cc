// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "kernel_context.h"

#ifdef USE_CUDA
#include "math/cuda/negpos_def.h"
#include <cuda.h>
#include <cuda_runtime.h>

class MockCudaKernelContext : public Ort::Custom::CUDAKernelContext {
public:
  MockCudaKernelContext() { cudaStreamCreate(&stream); }
  ~MockCudaKernelContext() { cudaStreamDestroy(stream); }
  void* AllocScratchBuffer(size_t size) override { return nullptr; }
  void FreeScratchBuffer(void* p) override {}
  void* AllocCudaScratchBuffer(size_t size) override { return nullptr; }
  void FreeCudaScratchBuffer(void* p) override {}
  void* GetCudaStream() const override { return static_cast<void*>(stream); }
  void* GetCublasHandle() const override { return nullptr; }
  int GetCudaDeviceId() const override { return 0; }

private:
  cudaStream_t stream;
};

class CudaAllocator : public Ort::Custom::IAllocator {
public:
  void* Alloc(size_t size) override { 
    void* p = nullptr;
    cudaMalloc((void**)&p, size);
    return p;
  }
  void Free(void* p) override { cudaFree(p); }
};

TEST(CudaOp, test_eager_negpos) {
  MockCudaKernelContext mock_cuda_kc;
  std::vector<float> input_data = {0.0f, 0.2f, -1.3f, 1.5f};
  std::unique_ptr<CudaAllocator> cuda_alloc = std::make_unique<CudaAllocator>();
  void* device_input = cuda_alloc->Alloc(sizeof(float) * input_data.size());
  cudaMemcpyAsync(device_input, input_data.data(), sizeof(float)*input_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  ortc::Tensor<float> input(std::vector<int64_t>{2, 2}, device_input);
  ortc::Tensor<float> output1(cuda_alloc.get());
  ortc::Tensor<float> output2(cuda_alloc.get());
  neg_pos_cuda(&mock_cuda_kc, input, output1, output2);

  float* host_output1 = (float*)malloc(sizeof(float) * input_data.size());
  float* host_output2 = (float*)malloc(sizeof(float) * input_data.size());
  cudaMemcpyAsync(host_output1, output1.DataRaw(), sizeof(float)*input_data.size(), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));
  cudaMemcpyAsync(host_output2, output2.DataRaw(), sizeof(float)*input_data.size(), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));
  ASSERT_NEAR(host_output1[1], input_data[1], 0.01f);
  ASSERT_NEAR(host_output2[2], input_data[2], 0.01f);
  ASSERT_NEAR(host_output1[3], input_data[3], 0.01f);

  cuda_alloc->Free(device_input);
  free(host_output1);
  free(host_output2);
}

#endif
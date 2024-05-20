// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "kernel_context.h"

#ifdef USE_CUDA
#include "math/cuda/negpos_def.h"
#include "cuda/fast_gelu.h"
#include <cuda.h>
#include <cuda_runtime.h>


class CudaAllocator : public Ort::Custom::IAllocator {
public:
  void* Alloc(size_t size) override { 
    void* p = nullptr;
    cudaMalloc((void**)&p, size);
    return p;
  }
  void Free(void* p) override { cudaFree(p); }
};

class MockCudaKernelContext : public Ort::Custom::CUDAKernelContext {
public:
  MockCudaKernelContext() { cudaStreamCreate(&stream); }
  ~MockCudaKernelContext() { cudaStreamDestroy(stream); }
  void* AllocScratchBuffer(size_t size) override { return malloc(size); }
  void FreeScratchBuffer(void* p) override { return free(p);}
  void* AllocCudaScratchBuffer(size_t size) override { return cuda_alloc.Alloc(size); }
  void FreeCudaScratchBuffer(void* p) override { return cuda_alloc.Free(p); }
  void* GetCudaStream() const override { return static_cast<void*>(stream); }
  void* GetCublasHandle() const override { return nullptr; }
  int GetCudaDeviceId() const override { return 0; }

  Ort::Custom::IAllocator* GetCudaAllocator() { return &cuda_alloc;};

private:
  cudaStream_t stream;
  CudaAllocator cuda_alloc;
};

TEST(CudaOp, test_eager_negpos) {
  MockCudaKernelContext mock_cuda_kc;
  std::vector<float> input_data = {0.0f, 0.2f, -1.3f, 1.5f};
  auto cuda_alloc = mock_cuda_kc.GetCudaAllocator();
  void* device_input = cuda_alloc->Alloc(sizeof(float) * input_data.size());
  cudaMemcpyAsync(device_input, input_data.data(), sizeof(float)*input_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  ortc::Tensor<float> input(std::vector<int64_t>{2, 2}, device_input);
  ortc::Tensor<float> output1(cuda_alloc);
  ortc::Tensor<float> output2(cuda_alloc);
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

TEST(CudaOp, test_fastgelu_eager) {

  MockCudaKernelContext mock_cuda_kc;
  // inputs
  std::vector<float> x_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  auto cuda_alloc = mock_cuda_kc.GetCudaAllocator();
  void* x_gpu_input = cuda_alloc->Alloc(sizeof(float) * x_data.size());
  cudaMemcpyAsync(x_gpu_input, x_data.data(), sizeof(float)*x_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  std::vector<float> bias_data = {0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
  void* bias_gpu_input = cuda_alloc->Alloc(sizeof(float) * bias_data.size());
  cudaMemcpyAsync(bias_gpu_input, bias_data.data(), sizeof(float)*bias_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  ortc::NamedArgumentDict dict({"use_half_2_"},
                               std::make_tuple(false));
  contrib::FastGelu<float> fast_gelu;  
  fast_gelu.OnModelAttach(dict);

  ortc::Tensor<float> x(std::vector<int64_t>{6, }, x_gpu_input);
  ortc::Tensor<float> bias(std::vector<int64_t>{6, }, bias_gpu_input);
  ortc::Tensor<float> output(cuda_alloc);
  fast_gelu.Compute(&mock_cuda_kc, x, &bias, output);

  float* host_output = (float*)malloc(sizeof(float) * x_data.size());
  cudaMemcpyAsync(host_output, output.DataRaw(), sizeof(float)*x_data.size(), cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));
  ASSERT_NEAR(host_output[1], 0.9505811, 0.01f);
  ASSERT_NEAR(host_output[2], 2.1696784, 0.01f);
  ASSERT_NEAR(host_output[3], 3.298689, 0.01f);
  ASSERT_NEAR(host_output[4], 4.399991, 0.01f);
  ASSERT_NEAR(host_output[5], 5.5, 0.01f);
}

#endif
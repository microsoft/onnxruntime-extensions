// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <locale>
#include "gtest/gtest.h"
#include "ocos.h"
#include "kernel_context.h"

#ifdef USE_CUDA
#include "math/cuda/negpos_def.h"
#include "cuda/attention_lib/group_query_attention.h"
#include "cuda/fast_gelu.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include <optional>


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

TEST(CudaOp, test_gqa_eager) {

  MockCudaKernelContext mock_cuda_kc;
  ortc::NamedArgumentDict dict({"num_heads", "kv_num_heads", "local_window_size", "rotary", "rotary_interleaved"},
                               std::make_tuple((int64_t)4, (int64_t)4, (int64_t)-1, (int64_t)false, (int64_t)false));
  contrib::GroupQueryAttention<ortc::MFloat16> GQA;
  GQA.OnModelAttach(dict);

  std::vector<float> query_fp32_data{-1.6592,  1.9277,  0.8760,  0.3105,  1.1377, -0.7349, -0.8086,
           0.5542,  0.4773, -0.7651, -0.3364,  0.8901, -1.6172, -1.3828,
           2.2129, -0.6030, -0.8359,  0.8130, -0.2239, -0.3994,  0.2673,
          -0.1252,  0.3840, -0.5801,  0.1830, -1.0537, -1.7383, -0.9712,
           0.2480, -1.3701,  0.7559, -0.5557};
  std::vector<ortc::MFloat16> query_data(query_fp32_data.begin(), query_fp32_data.end());

  std::vector<float> past_key_fp32_data{ 0.5010, -0.0542,  0.5386,  0.2764, -1.4385, -1.5312,  0.1119,  1.7080,
        -0.1099, -0.3079,  0.6372, -0.7539, -0.0911, -0.9551,  0.5029,  0.2251,
         1.3135,  2.0723,  1.2764, -0.2993,  1.6289, -0.5664, -1.5410,  0.8188,
         0.3479, -0.6240, -0.1943,  0.0476,  0.5396, -0.3943, -1.1904,  1.7070,
        -0.7700, -1.3760,  0.5176, -0.7925, -0.0111,  0.4668,  0.7832, -2.2246,
         1.0742, -0.0551, -0.3535, -2.1895,  0.6045, -0.1617,  1.8232,  0.5317,
        -0.2417,  0.6602,  0.1171,  2.5059, -0.8545,  1.5771,  0.7280, -0.6860,
         0.2258,  0.4800, -0.3633,  1.7559,  1.8066,  0.0654,  0.0540, -1.3291};
  std::vector<ortc::MFloat16> past_key_data(past_key_fp32_data.begin(), past_key_fp32_data.end());

  std::vector<float> past_value_fp32_data {-0.5835, -0.8921, -0.5298,  1.0850,  1.2051, -0.5659, -0.1124, -0.6567,
        -1.1182, -0.5957,  0.2952,  1.0215, -0.7632,  0.7295, -0.4319, -0.4116,
        -0.5938, -1.2607, -0.3037,  0.5249,  0.1610, -0.0620, -0.1490, -0.1721,
        -1.3164,  0.5884,  1.0400,  1.2471, -0.9409, -2.7012, -0.1023, -0.5967,
        -0.7583,  0.8965, -1.5752, -0.8535, -0.2247, -0.7705,  0.8159,  0.2113,
        -1.5742, -0.3538, -0.6343, -0.3789,  0.2079,  1.6826,  1.7314, -1.3691,
         0.4917,  0.7573,  0.5498, -0.3804, -0.0951, -0.8687, -2.8359, -0.5874,
        -0.9648,  0.2649, -0.0262,  0.5845,  0.3723,  1.0117,  0.3867, -2.3340};
  std::vector<ortc::MFloat16> past_value_data(past_value_fp32_data.begin(), past_value_fp32_data.end());

  std::vector<float> key_fp32_data {-0.9658, -0.2551, -0.3589,  0.7075,  0.5664, -0.8550, -1.8037, -0.0263,
        -2.0117,  1.2432, -0.1371, -0.6460,  1.6084, -0.7856,  0.3774,  0.0493,
        -1.9062,  1.6357,  1.6689,  0.6250, -0.9961, -1.1406, -0.5303, -0.5591,
        -0.2861, -1.4609, -0.3911,  0.9136,  0.4893,  0.1588,  0.5972, -0.9507};
  std::vector<ortc::MFloat16> key_data(key_fp32_data.begin(), key_fp32_data.end());

  std::vector<float> value_fp32_data {1.7578,  0.7573, -0.3792, -0.2634,  0.0267,  0.1066, -0.4268,  1.8516,
        -1.1758,  0.5981, -0.3325,  1.5234,  0.7876, -0.1825,  0.6123,  0.9810,
         0.2473,  1.1494,  1.4395, -0.8579,  1.0684, -0.4692, -0.1188, -1.5713,
        -1.5430, -2.5391,  0.8301, -0.3464, -0.3789, -2.0332, -2.0508, -0.3186};
  std::vector<ortc::MFloat16> value_data(value_fp32_data.begin(), value_fp32_data.end());

  std::vector<int32_t> seqlens_k = {2};
  std::vector<int32_t> total_sequence_length = {3};

  auto cuda_alloc = mock_cuda_kc.GetCudaAllocator();
  void* query_data_gpu = cuda_alloc->Alloc(sizeof(ortc::MFloat16) * query_data.size());
  cudaMemcpyAsync(query_data_gpu, query_data.data(), sizeof(ortc::MFloat16)*query_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  void* past_key_data_gpu = cuda_alloc->Alloc(sizeof(ortc::MFloat16) * past_key_data.size());
  cudaMemcpyAsync(past_key_data_gpu, past_key_data.data(), sizeof(ortc::MFloat16)*past_key_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));
  
  void* past_value_data_gpu = cuda_alloc->Alloc(sizeof(ortc::MFloat16) * past_value_data.size());
  cudaMemcpyAsync(past_value_data_gpu, past_value_data.data(), sizeof(ortc::MFloat16)*past_value_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  void* key_data_gpu = cuda_alloc->Alloc(sizeof(ortc::MFloat16) * key_data.size());
  cudaMemcpyAsync(key_data_gpu, key_data.data(), sizeof(ortc::MFloat16)*key_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  void* value_data_gpu = cuda_alloc->Alloc(sizeof(ortc::MFloat16) * value_data.size());
  cudaMemcpyAsync(value_data_gpu, value_data.data(), sizeof(ortc::MFloat16)*value_data.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));

  void* seqlens_k_data_gpu = cuda_alloc->Alloc(sizeof(int32_t));
  cudaMemcpyAsync(seqlens_k_data_gpu, seqlens_k.data(), sizeof(int32_t)*seqlens_k.size(), cudaMemcpyHostToDevice, static_cast<cudaStream_t>(mock_cuda_kc.GetCudaStream()));
  // input tensors
  ortc::Tensor<ortc::MFloat16> query(std::vector<int64_t>{1, 1, 32}, query_data_gpu);
  ortc::Tensor<ortc::MFloat16> key(std::vector<int64_t>{1, 1, 32}, key_data_gpu);
  ortc::Tensor<ortc::MFloat16> value(std::vector<int64_t>{1, 1, 32}, value_data_gpu);

  ortc::Tensor<ortc::MFloat16> past_key(std::vector<int64_t>{1, 4, 2, 8}, past_key_data_gpu);
  ortc::Tensor<ortc::MFloat16> past_value(std::vector<int64_t>{1, 4, 2, 8}, past_value_data_gpu);
  ortc::Tensor<int32_t> seqlens_k_gpu(std::vector<int64_t>{1,}, seqlens_k_data_gpu);
  ortc::Tensor<int32_t> total_sequence_cpu(std::vector<int64_t>{1,}, total_sequence_length.data());
  ortc::Tensor<ortc::MFloat16> output(cuda_alloc);

  auto status = GQA.Compute(&mock_cuda_kc, query, &key, &value, &past_key, &past_value, seqlens_k_gpu, total_sequence_cpu, std::nullopt, std::nullopt, output, std::nullopt, std::nullopt);

  cudaDeviceSynchronize();

  assert(status == nullptr);
}

#endif
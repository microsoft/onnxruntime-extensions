// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "exceptions.h"
#include "tensor_api.h"

namespace ortc = Ort::Custom;

class TestAllocator : public ortc::IAllocator {
public:
  void* Alloc(size_t size) override {
    return malloc(size);
  }
  void Free(void* p) override {
    if (p)
      free(p);
  }
};

TEST(tensor_api, test_move) {
  TestAllocator test_allocator;

  std::vector<float> input_data = {0.0f, 0.2f, -1.3f, 1.5f};
  
  ortc::Tensor<float> input(std::vector<int64_t>{2, 2}, input_data.data());

  ortc::Tensor<float> input_moved = std::move(input);
  const float* data = input_moved.Data();
  for (auto i = 0; i < 4; ++i)
   EXPECT_NEAR(data[i], input_data[i], 1e-5);

  ortc::Tensor<float> output1(&test_allocator);

  float* allocated_data = output1.Allocate(std::vector<int64_t>{2, 2});
  memcpy(allocated_data, input_data.data(), 4 * sizeof(float));

  ortc::Tensor<float> output_moved = std::move(output1);
  const float* output_data = output_moved.Data();
  EXPECT_EQ(output_data, allocated_data);  
}

TEST(tensor_api, test_release) {
  TestAllocator test_allocator;

  std::vector<float> input_data = {0.0f, 0.2f, -1.3f, 1.5f};
  
  ortc::Tensor<float> output(&test_allocator);

  float* allocated_data = output.Allocate(std::vector<int64_t>{2, 2});
  memcpy(allocated_data, input_data.data(), 4 * sizeof(float));

  void* released_buffer = output.Release();
  EXPECT_EQ(released_buffer, allocated_data);  
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "ortx_cpp_helper.h"
#include "shared/api/image_processor.h"

using namespace ort_extensions;

const char* test_image_paths[] = {"data/processor/standard_s.jpg", "data/processor/australia.jpg", "data/processor/exceltable.png"};
const size_t test_image_count = sizeof(test_image_paths) / sizeof(test_image_paths[0]);

TEST(ProcessorTest, TestPhi3VImageProcessing) {
  auto [input_data, n_data] = ort_extensions::LoadRawImages(test_image_paths, test_image_count);

  auto proc = OrtxObjectPtr<ImageProcessor>(OrtxCreateProcessor, "data/processor/phi_3_image.json");
  ortc::Tensor<float>* pixel_values;
  ortc::Tensor<int64_t>* image_sizes;
  ortc::Tensor<int64_t>* num_img_tokens;

  auto [status, r] = proc->PreProcess(ort_extensions::span(input_data.get(), (size_t)n_data), &pixel_values,
                                      &image_sizes, &num_img_tokens);

  ASSERT_TRUE(status.IsOk());
  int64_t expected_image_size[] = {1344, 1344, 1008, 1344, 1008, 1680};
  int64_t expected_num_token[] = {2509, 1921, 2353};

  ASSERT_EQ(pixel_values->Shape(), std::vector<int64_t>({3, 17, 3, 336, 336}));
  ASSERT_EQ(image_sizes->Shape(), std::vector<int64_t>({3, 2}));
  ASSERT_EQ(num_img_tokens->Shape(), std::vector<int64_t>({3, 1}));

  // compare the image sizes
  for (size_t i = 0; i < 3; i++) {
    ASSERT_EQ(image_sizes->Data()[i * 2], expected_image_size[i * 2]);
    ASSERT_EQ(image_sizes->Data()[i * 2 + 1], expected_image_size[i * 2 + 1]);
    ASSERT_EQ(num_img_tokens->Data()[i], expected_num_token[i]);
  }

  proc->ClearOutputs(&r);
}

TEST(ProcessorTest, TestCLIPImageProcessing) {
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), test_image_paths, test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/processor/clip_image.json");
  if (err != kOrtxOK) {
    std::cout << "Error: " << OrtxGetLastErrorMessage() << std::endl;
  }
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxTensor* tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, &tensor);
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor, reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 4);
}

TEST(ProcessorTest, TestMLlamaImageProcessing) {
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), test_image_paths, test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/processor/mllama/llama_3_image.json");
  if (err != kOrtxOK) {
    std::cout << "Error: " << OrtxGetLastErrorMessage() << std::endl;
  }
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 5);

  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const int64_t* int_data{};
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&int_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 2);
  ASSERT_EQ(std::vector<int64_t>(int_data, int_data + 3), std::vector<int64_t>({6, 6, 1}));

  err = OrtxTensorResultGetAt(result.get(), 2, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&int_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 2);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 4}));

  err = OrtxTensorResultGetAt(result.get(), 3, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&int_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 2);
  ASSERT_EQ(std::vector<int64_t>(int_data, int_data + 3), std::vector<int64_t>({4, 4, 1}));
}

TEST(ProcessorTest, TestPhi4VisionProcessor) {
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), test_image_paths, test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/models/phi-4/vision_processor.json");
  if (err != kOrtxOK) {
    std::cout << "Error: " << OrtxGetLastErrorMessage() << std::endl;
  }

  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  // embedding data (float32)
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const float* data{};
  const int64_t* int_data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 10, 3, 448, 448}));
  EXPECT_FLOAT_EQ(data[0], -0.295063f);

  // image sizes (int64_t)
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&int_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 2}));
  EXPECT_EQ(std::vector<int64_t>(int_data, int_data + 6),
            std::vector<int64_t>({1344, 1344, 896, 1344, 448, 896}));

  // mask data (float32)
  err = OrtxTensorResultGetAt(result.get(), 2, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 10, 32, 32}));
  EXPECT_FLOAT_EQ(data[0], 1.0f);

  // num tokens (int64_t)
  err = OrtxTensorResultGetAt(result.get(), 3, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&int_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(std::vector<int64_t>(shape, shape + num_dims), std::vector<int64_t>({3, 1}));
  EXPECT_EQ(std::vector<int64_t>(int_data, int_data + 3), std::vector<int64_t>({2625, 1841, 735}));
}

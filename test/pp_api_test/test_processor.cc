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


TEST(ProcessorTest, TestPhi3VImageProcessing) {
  auto [input_data, n_data] = ort_extensions::LoadRawImages(
      // {"data/processor/standard_s.jpg", "data/processor/australia.jpg", "data/processor/exceltable.png"});
      {"C:\\g\\phi-3-mini-v\\image.jpg", "data/processor/australia.jpg", "data/processor/exceltable.png"});

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

TEST(ProcessorTest, TestClipImageProcessing) {
  const char* images_path[] = {"data/processor/standard_s.jpg", "data/processor/australia.jpg",
                               "data/processor/exceltable.png"};
  OrtxObjectPtr<OrtxRawImages> raw_images;
  extError_t err = OrtxLoadImages(ort_extensions::ptr(raw_images), images_path, 3, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(ort_extensions::ptr(processor), "data/processor/clip_image.json");
  if (err != kOrtxOK) {
    std::cout << "Error: " << OrtxGetLastErrorMessage() << std::endl;
  }
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), ort_extensions::ptr(result));
  ASSERT_EQ(err, kOrtxOK);

  OrtxTensor* tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, &tensor);
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorDataFloat(tensor, &data, &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 4);
}

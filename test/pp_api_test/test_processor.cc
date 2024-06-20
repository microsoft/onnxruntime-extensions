// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <fstream>
#include <filesystem>

#include "gtest/gtest.h"
#include "ortx_c_helper.h"
#include "shared/api/image_processor.h"

using namespace ort_extensions;

std::vector<float> ReadArrayFromFile(const std::string& filename) {
  std::ifstream inFile(filename, std::ios::binary | std::ios::ate);
  if (!inFile) {
    throw std::runtime_error("Cannot open file for reading.");
  }
  std::streamsize fileSize = inFile.tellg();
  inFile.seekg(0, std::ios::beg);
  std::vector<float> array(fileSize / sizeof(float));
  if (!inFile.read(reinterpret_cast<char*>(array.data()), fileSize)) {
    throw std::runtime_error("Error reading file.");
  }

  return array;
}

TEST(ProcessorTest, TestPhi3VImageProcessing) {
  auto [input_data, n_data] = ort_extensions::LoadRawImages(
      {"data/processor/standard_s.jpg", "data/processor/australia.jpg", "data/processor/exceltable.png"});

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

  if (std::filesystem::is_directory("data2/processor")) {
    // the test data was dumped in this way
    // {
    // std::ofstream outFile("data2/processor/img_proc_pixel_values.bin", std::ios::binary);
    // outFile.write(reinterpret_cast<const char*>(array.data()), array.size() * sizeof(float));
    // }

    auto expected_output = ReadArrayFromFile("data2/processor/img_proc_pixel_values.bin");
    ASSERT_EQ(pixel_values->NumberOfElement(), expected_output.size());
    for (size_t i = 0; i < expected_output.size(); i++) {
      ASSERT_NEAR(pixel_values->Data()[i], expected_output[i], 1e-3);
    }
  }

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

  OrtxObjectPtr<OrtxImageProcessorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), ort_extensions::ptr(result));
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxImageGetTensorResult(result.get(), 0, ort_extensions::ptr(tensor));
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorDataFloat(tensor.get(), &data, &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 4);
}

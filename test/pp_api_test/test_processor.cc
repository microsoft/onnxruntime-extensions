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
  EXPECT_TRUE((data[0] > -0.30f) && (data[0] < -0.29f));

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

TEST(ProcessorTest, TestQwen2_5VLImageProcessing) {
  const char* qwen_test_image_path[] = {"data/processor/australia.jpg"};
  const size_t qwen_test_image_count = 1;

  // Load Image
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), qwen_test_image_path, qwen_test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  // Create Processor
  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/qwen2.5vl/vision_processor.json");
  ASSERT_EQ(err, kOrtxOK);

  // Run Processor
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Extract tensor
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* cpp_data{};
  const int64_t* shape{};
  size_t num_dims{};
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&cpp_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);

  // Expect 3 Dimensions
  ASSERT_EQ(num_dims, 3ULL);

  // Read the expected output from file
  std::filesystem::path expected_pixel_values_output_path = "data/qwen2.5vl/pixel_values_reference.txt";
  std::ifstream ref(expected_pixel_values_output_path);
  ASSERT_TRUE(ref.is_open()) << "Could not open reference output file.";

  std::vector<float> reference;
  reference.reserve(1000);

  std::string line;
  while (std::getline(ref, line)) {
    reference.push_back(std::stof(line));
  }

  ref.close();
  ASSERT_EQ(reference.size(), 1000) << "Reference float count does not match C++ output count.";

  // Compute MSE
  double mse = 0.0;
  for (size_t i = 0; i < 1000; i++) {

    double diff = static_cast<double>(cpp_data[i]) - static_cast<double>(reference[i]);
    mse += diff * diff;
  }
  mse /= 1000;

  ASSERT_LE(mse, 1e-3);
}

TEST(ProcessorTest, TestQwen3VLImageProcessing) {
  const char* qwen_test_image_path[] = {"data/processor/australia.jpg"};
  const size_t qwen_test_image_count = 1;

  // Load Image
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), qwen_test_image_path, qwen_test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  // Create Processor with Qwen3-VL config (patch_size=16 vs 14 for Qwen2.5-VL)
  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/qwen3vl/vision_processor.json");
  ASSERT_EQ(err, kOrtxOK);

  // Run Processor
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Extract pixel_values tensor
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* cpp_data{};
  const int64_t* shape{};
  size_t num_dims{};
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&cpp_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);

  // Expect 3 Dimensions: [num_patches, patch_dim]
  ASSERT_EQ(num_dims, 3ULL);

  // Qwen3-VL uses patch_size=16, so patch_dim = 3 * 2 * 16 * 16 = 1536
  // (vs Qwen2.5-VL's 3 * 2 * 14 * 14 = 1176)
  int64_t patch_dim = shape[2];
  ASSERT_EQ(patch_dim, 1536);
}

// Security regression test: CMYK JPEG must be rejected at decode time (CWE-122).
//
// A CMYK JPEG has 4 output channels. Phi4VisionDynamicPreprocess allocates a
// 3-channel output buffer but previously copied using the dynamic channel count,
// writing past the buffer end and enabling heap corruption / RCE.
// Both the JPEG decoder and the transform ops now validate channel count.
TEST(ProcessorTest, TestCMYKJpegRejected) {
  // Minimal 4x4 CMYK JPEG generated by PIL (Image.new("CMYK", (4,4), (128,64,32,255)))
  // Contains an APP14 Adobe marker indicating CMYK color space.
  static const uint8_t cmyk_jpeg[] = {
    0xff,0xd8,0xff,0xee,0x00,0x0e,0x41,0x64,0x6f,0x62,0x65,0x00,0x64,0x00,0x00,0x00,
    0x00,0x00,0xff,0xdb,0x00,0x43,0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,
    0x07,0x09,0x09,0x08,0x0a,0x0c,0x14,0x0d,0x0c,0x0b,0x0b,0x0c,0x19,0x12,0x13,0x0f,
    0x14,0x1d,0x1a,0x1f,0x1e,0x1d,0x1a,0x1c,0x1c,0x20,0x24,0x2e,0x27,0x20,0x22,0x2c,
    0x23,0x1c,0x1c,0x28,0x37,0x29,0x2c,0x30,0x31,0x34,0x34,0x34,0x1f,0x27,0x39,0x3d,
    0x38,0x32,0x3c,0x2e,0x33,0x34,0x32,0xff,0xc0,0x00,0x14,0x08,0x00,0x04,0x00,0x04,
    0x04,0x43,0x11,0x00,0x4d,0x11,0x00,0x59,0x11,0x00,0x4b,0x11,0x00,0xff,0xc4,0x00,
    0x1f,0x00,0x00,0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,0x00,
    0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,0x09,0x0a,0x0b,0xff,0xc4,
    0x00,0xb5,0x10,0x00,0x02,0x01,0x03,0x03,0x02,0x04,0x03,0x05,0x05,0x04,0x04,0x00,
    0x00,0x01,0x7d,0x01,0x02,0x03,0x00,0x04,0x11,0x05,0x12,0x21,0x31,0x41,0x06,0x13,
    0x51,0x61,0x07,0x22,0x71,0x14,0x32,0x81,0x91,0xa1,0x08,0x23,0x42,0xb1,0xc1,0x15,
    0x52,0xd1,0xf0,0x24,0x33,0x62,0x72,0x82,0x09,0x0a,0x16,0x17,0x18,0x19,0x1a,0x25,
    0x26,0x27,0x28,0x29,0x2a,0x34,0x35,0x36,0x37,0x38,0x39,0x3a,0x43,0x44,0x45,0x46,
    0x47,0x48,0x49,0x4a,0x53,0x54,0x55,0x56,0x57,0x58,0x59,0x5a,0x63,0x64,0x65,0x66,
    0x67,0x68,0x69,0x6a,0x73,0x74,0x75,0x76,0x77,0x78,0x79,0x7a,0x83,0x84,0x85,0x86,
    0x87,0x88,0x89,0x8a,0x92,0x93,0x94,0x95,0x96,0x97,0x98,0x99,0x9a,0xa2,0xa3,0xa4,
    0xa5,0xa6,0xa7,0xa8,0xa9,0xaa,0xb2,0xb3,0xb4,0xb5,0xb6,0xb7,0xb8,0xb9,0xba,0xc2,
    0xc3,0xc4,0xc5,0xc6,0xc7,0xc8,0xc9,0xca,0xd2,0xd3,0xd4,0xd5,0xd6,0xd7,0xd8,0xd9,
    0xda,0xe1,0xe2,0xe3,0xe4,0xe5,0xe6,0xe7,0xe8,0xe9,0xea,0xf1,0xf2,0xf3,0xf4,0xf5,
    0xf6,0xf7,0xf8,0xf9,0xfa,0xff,0xda,0x00,0x0e,0x04,0x43,0x00,0x4d,0x00,0x59,0x00,
    0x4b,0x00,0x00,0x3f,0x00,0x4a,0xef,0xeb,0xd7,0xeb,0xe7,0xfa,0xff,0xd9
  };

  // Write CMYK JPEG to a temp file
  std::filesystem::path temp_path = std::filesystem::temp_directory_path() / "cmyk_test_security.jpg";
  {
    std::ofstream f(temp_path, std::ios::binary);
    ASSERT_TRUE(f.is_open()) << "Failed to create temp CMYK JPEG file";
    f.write(reinterpret_cast<const char*>(cmyk_jpeg), sizeof(cmyk_jpeg));
  }

  const std::string cmyk_path_string = temp_path.string();
  const char* cmyk_path_str = cmyk_path_string.c_str();

  // Attempt to load the CMYK JPEG - should be rejected by the decoder
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), &cmyk_path_str, 1, nullptr);

  if (err == kOrtxOK) {
    // If the image loaded (shouldn't with our decoder fix), verify the processor rejects it
    OrtxObjectPtr<OrtxProcessor> processor;
    err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/models/phi-4/vision_processor.json");
    if (err == kOrtxOK) {
      OrtxObjectPtr<OrtxTensorResult> result;
      err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
    }
  }

  // Clean up temp file
  std::filesystem::remove(temp_path);

  // The CMYK JPEG must be rejected somewhere in the pipeline (CWE-122 mitigation)
  ASSERT_NE(err, kOrtxOK) << "CMYK JPEG must be rejected to prevent heap buffer overflow (CWE-122)";
}


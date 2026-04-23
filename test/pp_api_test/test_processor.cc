// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <tuple>
#include <algorithm>
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

TEST(ProcessorTest, TestGemma4ImageProcessing) {
  // Use a single test image to verify the Gemma 4 vision preprocessing pipeline:
  // DecodeImage -> Gemma4ImageTransform (aspect-ratio resize + patchify + position IDs)
  const char* image_path[] = {"data/processor/australia.jpg"};

  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), image_path, 1, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/models/gemma-4/image_processor.json");
  if (err != kOrtxOK) {
    std::cout << "Error: " << OrtxGetLastErrorMessage() << std::endl;
  }
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output 0: pixel_values — float (batch, max_patches, patch_dim)
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* pv_data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&pv_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);           // (batch, max_patches, patch_dim)
  ASSERT_EQ(shape[0], 1);              // single image
  // Default max_patches = 280 * 9 = 2520
  constexpr int64_t kMaxPatches = 280 * 9;
  constexpr int64_t kPatchDim = 16 * 16 * 3;  // 768
  ASSERT_EQ(shape[1], kMaxPatches);
  ASSERT_EQ(shape[2], kPatchDim);

  // Pixel values should be rescaled to [0, 1].
  bool all_in_range = true;
  for (int64_t i = 0; i < std::min<int64_t>(shape[1] * shape[2], 10000); ++i) {
    if (pv_data[i] < 0.0f || pv_data[i] > 1.0f) {
      all_in_range = false;
      break;
    }
  }
  EXPECT_TRUE(all_in_range) << "Pixel values should be in [0, 1]";

  // Verify pixel values match HuggingFace Gemma4ImageProcessor output.
  // Reference: HF transformers Gemma4ImageProcessor on australia.jpg (1300x876).
  // Patch 0, first 10 values (HWC order within each patch):
  const float kPatch0Expected[] = {
      0.18823531f, 0.05490196f, 0.01960784f,
      0.18823531f, 0.05490196f, 0.01960784f,
      0.18823531f, 0.05490196f, 0.01960784f,
      0.18823531f};
  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(pv_data[i], kPatch0Expected[i], 1e-3f)
        << "Patch 0, value " << i << " mismatch vs HF reference";
  }
  // Patch 1, first 10 values:
  const float kPatch1Expected[] = {
      0.18039216f, 0.04705883f, 0.01176471f,
      0.18039216f, 0.04705883f, 0.01176471f,
      0.17647059f, 0.04313726f, 0.00784314f,
      0.17647059f};
  const float* patch1 = pv_data + kPatchDim;  // start of patch 1
  for (int i = 0; i < 10; ++i) {
    EXPECT_NEAR(patch1[i], kPatch1Expected[i], 1e-3f)
        << "Patch 1, value " << i << " mismatch vs HF reference";
  }

  // Output 1: position_ids — int64 (batch, max_patches, 2)
  err = OrtxTensorResultGetAt(result.get(), 1, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const int64_t* pos_data{};
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&pos_data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);           // (batch, max_patches, 2)
  ASSERT_EQ(shape[1], kMaxPatches);
  ASSERT_EQ(shape[2], 2);

  // Verify position IDs match HF reference.
  // Patch 0: (x=0, y=0), Patch 1: (x=1, y=0)  — HF uses meshgrid(arange(pw), arange(ph), indexing="xy")
  EXPECT_EQ(pos_data[0], 0);  // patch 0 x
  EXPECT_EQ(pos_data[1], 0);  // patch 0 y
  EXPECT_EQ(pos_data[2], 1);  // patch 1 x
  EXPECT_EQ(pos_data[3], 0);  // patch 1 y

  // Derive the expected number of real patches from num_soft_tokens.
  // Each soft token maps to pooling_kernel_size^2 = 9 patches.
  // Read num_soft_tokens early (output 2) to compute expected real patch count.
  OrtxObjectPtr<OrtxTensor> nst_tensor;
  err = OrtxTensorResultGetAt(result.get(), 2, nst_tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);
  const int64_t* nst_peek{};
  const int64_t* nst_shape_peek{};
  size_t nst_dims_peek{};
  err = OrtxGetTensorData(nst_tensor.get(), reinterpret_cast<const void**>(&nst_peek), &nst_shape_peek, &nst_dims_peek);
  ASSERT_EQ(err, kOrtxOK);
  int64_t expected_real_patches = nst_peek[0] * 9;  // pooling_kernel_size^2
  ASSERT_GT(expected_real_patches, 0);
  ASSERT_LE(expected_real_patches, kMaxPatches);

  // Verify that positions beyond the real patches are (-1, -1) padding.
  for (int64_t i = expected_real_patches; i < kMaxPatches; ++i) {
    EXPECT_EQ(pos_data[i * 2], -1) << "Padding position " << i << " x should be -1";
    EXPECT_EQ(pos_data[i * 2 + 1], -1) << "Padding position " << i << " y should be -1";
  }

  // Verify last real patch position matches HF: (59, 38)
  EXPECT_EQ(pos_data[(expected_real_patches - 1) * 2], 59);
  EXPECT_EQ(pos_data[(expected_real_patches - 1) * 2 + 1], 38);

  // Output 2: num_soft_tokens — verify exact value from HF reference (260).
  ASSERT_EQ(nst_dims_peek, 2ULL);
  ASSERT_EQ(nst_shape_peek[0], 1);
  EXPECT_EQ(nst_peek[0], 260) << "num_soft_tokens should be 260 for australia.jpg (HF reference)";
}

TEST(ProcessorTest, TestGemma4ImageProcessingMultiImage) {
  // Verify batched processing works with multiple images of different sizes.
  const char* image_paths[] = {"data/processor/standard_s.jpg", "data/processor/australia.jpg"};

  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), image_paths, 2, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/models/gemma-4/image_processor.json");
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // pixel_values batch dim should be 2
  OrtxObjectPtr<OrtxTensor> tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* data{};
  const int64_t* shape{};
  size_t num_dims;
  err = OrtxGetTensorData(tensor.get(), reinterpret_cast<const void**>(&data), &shape, &num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(num_dims, 3ULL);
  ASSERT_EQ(shape[0], 2);  // batch of 2 images
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

TEST(ProcessorTest, TestPixtralImageProcessingSingleImage) {
  const char* test_image_path[] = {"data/processor/australia.jpg"};
  const size_t test_image_count = 1;

  // Load image
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), test_image_path, test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  // Create processor with Pixtral config
  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/pixtral/vision_processor.json");
  ASSERT_EQ(err, kOrtxOK);

  // Run processor
  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output[0]: pixel_values — should be [N, C, H, W] where N=1
  OrtxObjectPtr<OrtxTensor> pixel_values_tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, pixel_values_tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* pixel_data{};
  const int64_t* pv_shape{};
  size_t pv_num_dims{};
  err = OrtxGetTensorData(pixel_values_tensor.get(), reinterpret_cast<const void**>(&pixel_data), &pv_shape, &pv_num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(pv_num_dims, 4ULL);  // [N, C, H, W]
  ASSERT_EQ(pv_shape[0], 1);     // single image
  ASSERT_EQ(pv_shape[1], 3);     // RGB channels

  // Output[1]: image_sizes — should be [N, 2] = [1, 2]
  OrtxObjectPtr<OrtxTensor> image_sizes_tensor;
  err = OrtxTensorResultGetAt(result.get(), 1, image_sizes_tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const int64_t* sizes_data{};
  const int64_t* is_shape{};
  size_t is_num_dims{};
  err = OrtxGetTensorData(image_sizes_tensor.get(), reinterpret_cast<const void**>(&sizes_data), &is_shape, &is_num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(is_num_dims, 2ULL);  // [N, 2]
  ASSERT_EQ(is_shape[0], 1);     // single image
  ASSERT_EQ(is_shape[1], 2);     // [H, W]

  // image_sizes should match actual pixel_values dimensions
  ASSERT_EQ(sizes_data[0], pv_shape[2]);  // H
  ASSERT_EQ(sizes_data[1], pv_shape[3]);  // W

  // H and W should be multiples of 28 (patch_size * merge_size) from smart resize
  ASSERT_EQ(sizes_data[0] % 28, 0);
  ASSERT_EQ(sizes_data[1] % 28, 0);
}

// Helper: run Pixtral processor on a single image and return pixel_values shape + data copy.
static void RunPixtralSingleImage(const char* image_path,
                                  std::vector<float>& out_pixels,
                                  int64_t out_shape[4],
                                  int64_t out_hw[2]) {
  OrtxObjectPtr<OrtxRawImages> raw{};
  ASSERT_EQ(OrtxLoadImages(raw.ToBeAssigned(), &image_path, 1, nullptr), kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> proc;
  ASSERT_EQ(OrtxCreateProcessor(proc.ToBeAssigned(), "data/pixtral/vision_processor.json"), kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> res;
  ASSERT_EQ(OrtxImagePreProcess(proc.get(), raw.get(), res.ToBeAssigned()), kOrtxOK);

  // pixel_values [1, C, H, W]
  OrtxObjectPtr<OrtxTensor> pv;
  ASSERT_EQ(OrtxTensorResultGetAt(res.get(), 0, pv.ToBeAssigned()), kOrtxOK);
  const float* data{};
  const int64_t* shape{};
  size_t ndims{};
  ASSERT_EQ(OrtxGetTensorData(pv.get(), reinterpret_cast<const void**>(&data), &shape, &ndims), kOrtxOK);
  ASSERT_EQ(ndims, 4ULL);
  size_t n = static_cast<size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
  out_pixels.assign(data, data + n);
  for (int i = 0; i < 4; ++i) out_shape[i] = shape[i];

  // image_sizes [1, 2]
  OrtxObjectPtr<OrtxTensor> is;
  ASSERT_EQ(OrtxTensorResultGetAt(res.get(), 1, is.ToBeAssigned()), kOrtxOK);
  const int64_t* sdata{};
  const int64_t* sshape{};
  size_t sndims{};
  ASSERT_EQ(OrtxGetTensorData(is.get(), reinterpret_cast<const void**>(&sdata), &sshape, &sndims), kOrtxOK);
  out_hw[0] = sdata[0];
  out_hw[1] = sdata[1];
}

TEST(ProcessorTest, TestPixtralImageProcessingMultiImage) {
  // Use two different-sized images to exercise padded-batch behavior
  const char* test_image_paths[] = {"data/processor/australia.jpg", "data/processor/standard_s.jpg"};
  const size_t test_image_count = 2;

  // --- Run each image individually for reference ---
  std::vector<float> single_pixels_0, single_pixels_1;
  int64_t single_shape_0[4], single_shape_1[4];
  int64_t single_hw_0[2], single_hw_1[2];
  RunPixtralSingleImage(test_image_paths[0], single_pixels_0, single_shape_0, single_hw_0);
  RunPixtralSingleImage(test_image_paths[1], single_pixels_1, single_shape_1, single_hw_1);

  // Images must have different dimensions for this test to be meaningful
  ASSERT_TRUE(single_hw_0[0] != single_hw_1[0] || single_hw_0[1] != single_hw_1[1])
      << "Test images should have different resized dimensions";

  // --- Run batch of two images ---
  OrtxObjectPtr<OrtxRawImages> raw_images{};
  extError_t err = OrtxLoadImages(raw_images.ToBeAssigned(), test_image_paths, test_image_count, nullptr);
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxProcessor> processor;
  err = OrtxCreateProcessor(processor.ToBeAssigned(), "data/pixtral/vision_processor.json");
  ASSERT_EQ(err, kOrtxOK);

  OrtxObjectPtr<OrtxTensorResult> result;
  err = OrtxImagePreProcess(processor.get(), raw_images.get(), result.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  // Output[0]: pixel_values [N, C, max_H, max_W]
  OrtxObjectPtr<OrtxTensor> pixel_values_tensor;
  err = OrtxTensorResultGetAt(result.get(), 0, pixel_values_tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const float* pixel_data{};
  const int64_t* pv_shape{};
  size_t pv_num_dims{};
  err = OrtxGetTensorData(pixel_values_tensor.get(), reinterpret_cast<const void**>(&pixel_data), &pv_shape, &pv_num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(pv_num_dims, 4ULL);
  ASSERT_EQ(pv_shape[0], 2);  // two images
  ASSERT_EQ(pv_shape[1], 3);  // RGB

  int64_t max_H = std::max(single_hw_0[0], single_hw_1[0]);
  int64_t max_W = std::max(single_hw_0[1], single_hw_1[1]);
  ASSERT_EQ(pv_shape[2], max_H);
  ASSERT_EQ(pv_shape[3], max_W);

  // Output[1]: image_sizes [N, 2]
  OrtxObjectPtr<OrtxTensor> image_sizes_tensor;
  err = OrtxTensorResultGetAt(result.get(), 1, image_sizes_tensor.ToBeAssigned());
  ASSERT_EQ(err, kOrtxOK);

  const int64_t* sizes_data{};
  const int64_t* is_shape{};
  size_t is_num_dims{};
  err = OrtxGetTensorData(image_sizes_tensor.get(), reinterpret_cast<const void**>(&sizes_data), &is_shape, &is_num_dims);
  ASSERT_EQ(err, kOrtxOK);
  ASSERT_EQ(is_num_dims, 2ULL);
  ASSERT_EQ(is_shape[0], 2);
  ASSERT_EQ(is_shape[1], 2);

  // image_sizes rows must match per-image dimensions
  ASSERT_EQ(sizes_data[0], single_hw_0[0]);  // H of image 0
  ASSERT_EQ(sizes_data[1], single_hw_0[1]);  // W of image 0
  ASSERT_EQ(sizes_data[2], single_hw_1[0]);  // H of image 1
  ASSERT_EQ(sizes_data[3], single_hw_1[1]);  // W of image 1

  // Validate pixel values: batch slice [i, :, :Hi, :Wi] must equal single-image output
  int64_t C = pv_shape[1];
  auto batch_idx = [&](int64_t n, int64_t c, int64_t h, int64_t w) {
    return n * C * max_H * max_W + c * max_H * max_W + h * max_W + w;
  };
  auto single_idx = [](int64_t c, int64_t h, int64_t w, int64_t sH, int64_t sW) {
    return c * sH * sW + h * sW + w;
  };

  // Check image 0: unpadded region matches single-image output
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t h = 0; h < single_hw_0[0]; ++h) {
      for (int64_t w = 0; w < single_hw_0[1]; ++w) {
        ASSERT_FLOAT_EQ(pixel_data[batch_idx(0, c, h, w)],
                        single_pixels_0[static_cast<size_t>(single_idx(c, h, w, single_hw_0[0], single_hw_0[1]))])
            << "Mismatch at image 0, c=" << c << " h=" << h << " w=" << w;
      }
    }
  }

  // Check image 1: unpadded region matches single-image output
  for (int64_t c = 0; c < C; ++c) {
    for (int64_t h = 0; h < single_hw_1[0]; ++h) {
      for (int64_t w = 0; w < single_hw_1[1]; ++w) {
        ASSERT_FLOAT_EQ(pixel_data[batch_idx(1, c, h, w)],
                        single_pixels_1[static_cast<size_t>(single_idx(c, h, w, single_hw_1[0], single_hw_1[1]))])
            << "Mismatch at image 1, c=" << c << " h=" << h << " w=" << w;
      }
    }
  }

  // Check padding is zero for the smaller image (image 0 if it has smaller H or W)
  // Bottom padding: rows [H0, max_H) for image 0
  if (single_hw_0[0] < max_H) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t h = single_hw_0[0]; h < max_H; ++h) {
        for (int64_t w = 0; w < max_W; ++w) {
          ASSERT_FLOAT_EQ(pixel_data[batch_idx(0, c, h, w)], 0.0f)
              << "Expected zero padding at image 0, c=" << c << " h=" << h << " w=" << w;
        }
      }
    }
  }
  // Right padding: cols [W0, max_W) for image 0
  if (single_hw_0[1] < max_W) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t h = 0; h < single_hw_0[0]; ++h) {
        for (int64_t w = single_hw_0[1]; w < max_W; ++w) {
          ASSERT_FLOAT_EQ(pixel_data[batch_idx(0, c, h, w)], 0.0f)
              << "Expected zero padding at image 0, c=" << c << " h=" << h << " w=" << w;
        }
      }
    }
  }
}


// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "OrtClient.h"
#import <Foundation/Foundation.h>

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_extensions.h>

@implementation OrtClient

// Runs a model and checks the result.
// Uses the ORT C++ API.
+ (BOOL)decodeAndCheckImageWithError:(NSError **)error {
  try {
    const auto ort_log_level = ORT_LOGGING_LEVEL_INFO;
    auto ort_env = Ort::Env(ort_log_level, "OrtExtensionsUsage");
    auto session_options = Ort::SessionOptions();

    if (RegisterCustomOps(session_options, OrtGetApiBase()) != nullptr) {
      throw std::runtime_error("RegisterCustomOps failed");
    }

    NSString *model_path = [NSBundle.mainBundle pathForResource:@"decode_image"
                                                         ofType:@"onnx"];
    if (model_path == nullptr) {
      throw std::runtime_error("Failed to get model path");
    }

    auto sess = Ort::Session(ort_env, [model_path UTF8String], session_options);

    // note: need to set Xcode settings to prevent it from messing with PNG files:
    // in "Build Settings":
    // - set "Compress PNG Files" to "No"
    // - set "Remove Text Metadata From PNG Files" to "No"
    NSString *input_image_path =
        [NSBundle.mainBundle pathForResource:@"r32_g64_b128_32x32" ofType:@"png"];
    if (input_image_path == nullptr) {
      throw std::runtime_error("Failed to get image path");
    }

    NSMutableData *input_data =
        [NSMutableData dataWithContentsOfFile:input_image_path];
    const int64_t input_data_length = input_data.length;
    const auto memoryInfo =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const auto input_tensor = Ort::Value::CreateTensor(
        memoryInfo, [input_data mutableBytes], input_data_length,
        &input_data_length, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8);

    constexpr auto input_names = std::array{"image"};
    constexpr auto output_names = std::array{"bgr_data"};

    const auto outputs = sess.Run(Ort::RunOptions(), input_names.data(),
                                  &input_tensor, 1, output_names.data(), 1);
    if (outputs.size() != 1) {
      throw std::runtime_error("Unexpected number of outputs");
    }

    const auto &output_tensor = outputs.front();
    const auto output_type_and_shape_info =
        output_tensor.GetTensorTypeAndShapeInfo();

    // We expect the model output to be BGR values (3 uint8's) for each pixel
    // in the decoded image.
    // The input image has 32x32 pixels.
    const int64_t h = 32, w = 32, c = 3;
    const std::vector<int64_t> expected_output_shape{h, w, c};
    const auto output_shape = output_type_and_shape_info.GetShape();
    if (output_shape != expected_output_shape) {
      throw std::runtime_error("Unexpected output shape");
    }

    if (const auto output_element_type =
            output_type_and_shape_info.GetElementType();
        output_element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) {
      throw std::runtime_error("Unexpected output element type");
    }

    // Each pixel in the input image has an RGB value of [32, 64, 128], or
    // equivalently, a BGR value of [128, 64, 32].
    const uint8_t expected_pixel_bgr_data[] = {128, 64, 32};
    const uint8_t *output_data_raw = output_tensor.GetTensorData<uint8_t>();
    for (size_t i = 0; i < h * w * c; ++i) {
      if (output_data_raw[i] != expected_pixel_bgr_data[i % 3]) {
        throw std::runtime_error("Unexpected pixel data");
      }
    }
  } catch (std::exception &e) {
    NSLog(@"%s error: %s", __FUNCTION__, e.what());
      
    static NSString *const kErrorDomain = @"OrtExtensionsUsage";
    constexpr NSInteger kErrorCode = 0;
    if (error) {
      NSString *description =
          [NSString stringWithCString:e.what() encoding:NSASCIIStringEncoding];
      *error =
          [NSError errorWithDomain:kErrorDomain
                              code:kErrorCode
                          userInfo:@{NSLocalizedDescriptionKey : description}];
    }
    return NO;
  }

  if (error) {
    *error = nullptr;
  }
  return YES;
}

@end

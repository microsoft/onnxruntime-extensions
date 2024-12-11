// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import Foundation
import CoreServices

enum OrtSwiftClientError: Error {
  case error(_ message: String)
}

// Runs a model and checks the result.
// Uses the ORT Objective-C/Swift API.
func swiftDecodeAndCheckImage() throws {
  let ort_log_level = ORTLoggingLevel.info
  let ort_env = try ORTEnv(loggingLevel: ort_log_level)

  let ort_session_options = try ORTSessionOptions()
  try ort_session_options.registerCustomOps(functionPointer: RegisterCustomOps)

  guard let model_path = Bundle.main.path(forResource: "decode_image", ofType: "onnx") else {
    throw OrtSwiftClientError.error("Failed to get model path")
  }

  let ort_session = try ORTSession(
    env: ort_env, modelPath: model_path, sessionOptions: ort_session_options)

  // note: need to set Xcode settings to prevent it from messing with PNG files:
  // in "Build Settings":
  // - set "Compress PNG Files" to "No"
  // - set "Remove Text Metadata From PNG Files" to "No"
  guard
    let input_image_url = Bundle.main.url(forResource: "r32_g64_b128_32x32", withExtension: "png")
  else {
    throw OrtSwiftClientError.error("Failed to get image URL")
  }

  let input_data = try Data(contentsOf: input_image_url)
  let input_data_length = input_data.count
  let input_shape = [NSNumber(integerLiteral: input_data_length)]
  let input_tensor = try ORTValue(
    tensorData: NSMutableData(data: input_data), elementType: ORTTensorElementDataType.uInt8,
    shape: input_shape)

  let outputs = try ort_session.run(
    withInputs: ["image": input_tensor], outputNames: ["bgr_data"], runOptions: nil)

  guard let output_tensor = outputs["bgr_data"] else {
    throw OrtSwiftClientError.error("Failed to get output")
  }

  let output_type_and_shape = try output_tensor.tensorTypeAndShapeInfo()

  // We expect the model output to be BGR values (3 uint8's) for each pixel
  // in the decoded image.
  // The input image has 32x32 pixels.
  let expected_output_shape: [NSNumber] = [32, 32, 3]
  guard output_type_and_shape.shape == expected_output_shape else {
    throw OrtSwiftClientError.error("Unexpected output shape")
  }

  let expected_output_element_type = ORTTensorElementDataType.uInt8
  guard output_type_and_shape.elementType == expected_output_element_type else {
    throw OrtSwiftClientError.error("Unexpected output element type")
  }

  // Each pixel in the input image has an RGB value of [32, 64, 128], or
  // equivalently, a BGR value of [128, 64, 32].
  let output_data: Data = try output_tensor.tensorData() as Data
  let expected_pixel_bgr_data: [UInt8] = [128, 64, 32]
  for (idx, byte) in output_data.enumerated() {
    guard byte == expected_pixel_bgr_data[idx % expected_pixel_bgr_data.count] else {
      throw OrtSwiftClientError.error("Unexpected pixel data")
    }
  }
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <cmath>
#include <algorithm>
#include "farmhash.h"
#include "string_tensor.h"
#include "string_hash.hpp"


KernelStringHash::KernelStringHash(OrtApi api) : BaseKernel(api) {
}

void KernelStringHash::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* num_buckets = ort_.KernelContext_GetInput(context, 1);
  const int64_t* p_num_buckets = ort_.GetTensorData<int64_t>(num_buckets);
  std::vector<std::string> str_input;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);

  // Verifications
  OrtTensorDimensions num_buckets_dimensions(ort_, num_buckets);
  if (num_buckets_dimensions.size() != 1 || num_buckets_dimensions[0] != 1)
    ORT_CXX_API_THROW(MakeString(
        "num_buckets must contain only one element. It has ",
        num_buckets_dimensions.size(), " dimensions."), ORT_INVALID_ARGUMENT);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  int64_t* out = ort_.GetTensorMutableData<int64_t>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  size_t nb = static_cast<size_t>(*p_num_buckets);
  for (int64_t i = 0; i < size; i++) {
    out[i] = static_cast<int64_t>(Hash64(str_input[i].c_str(), str_input[i].size()) % nb);
  }
}

void* CustomOpStringHash::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringHash(api);
};

const char* CustomOpStringHash::GetName() const { return "StringToHashBucket"; };

size_t CustomOpStringHash::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringHash::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
};

size_t CustomOpStringHash::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringHash::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

KernelStringHashFast::KernelStringHashFast(OrtApi api) : BaseKernel(api) {
}

void KernelStringHashFast::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  const OrtValue* num_buckets = ort_.KernelContext_GetInput(context, 1);
  const int64_t* p_num_buckets = ort_.GetTensorData<int64_t>(num_buckets);
  std::vector<std::string> str_input;
  GetTensorMutableDataString(api_, ort_, context, input, str_input);

  // Verifications
  OrtTensorDimensions num_buckets_dimensions(ort_, num_buckets);
  if (num_buckets_dimensions.size() != 1 || num_buckets_dimensions[0] != 1)
    ORT_CXX_API_THROW(MakeString(
        "num_buckets must contain only one element. It has ",
        num_buckets_dimensions.size(), " dimensions."), ORT_INVALID_ARGUMENT);

  // Setup output
  OrtTensorDimensions dimensions(ort_, input);
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  int64_t* out = ort_.GetTensorMutableData<int64_t>(output);

  OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  int64_t size = ort_.GetTensorShapeElementCount(output_info);
  ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  // Do computation
  size_t nb = static_cast<size_t>(*p_num_buckets);
  for (int64_t i = 0; i < size; i++) {
    out[i] = static_cast<int64_t>(util::Fingerprint64(str_input[i].c_str(), str_input[i].size()) % nb);
  }
}

void* CustomOpStringHashFast::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelStringHashFast(api);
};

const char* CustomOpStringHashFast::GetName() const { return "StringToHashBucketFast"; };

size_t CustomOpStringHashFast::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpStringHashFast::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      ORT_CXX_API_THROW(MakeString("Unexpected input index ", index), ORT_INVALID_ARGUMENT);
  }
};

size_t CustomOpStringHashFast::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringHashFast::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

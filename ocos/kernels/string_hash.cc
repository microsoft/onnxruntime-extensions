// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "string_hash.hpp"
#include <vector>
#include <cmath>
#include <algorithm>
#include "re2/re2.h"
#include "farmhash.h"
#include "string_common.h"

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hash.cc#L28
static inline uint64_t ByteAs64(char c) { return static_cast<uint64_t>(c) & 0xff; }

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/raw_coding.h#L41
uint64_t DecodeFixed32(const char* ptr) {
  return ((static_cast<uint64_t>(static_cast<unsigned char>(ptr[0]))) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[1])) << 8) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[2])) << 16) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[3])) << 24));
}

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/raw_coding.h#L55
static uint64_t DecodeFixed64(const char* ptr) {
  uint64_t lo = DecodeFixed32(ptr);
  uint64_t hi = DecodeFixed32(ptr + 4);
  return (hi << 32) | lo;
}

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hash.cc#L79
uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (n * m);

  while (n >= 8) {
    uint64_t k = DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64_t Hash64Fast(const char* data, size_t n) {
  return static_cast<int64_t>(util::Fingerprint64(data, n));
}

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
    throw std::runtime_error(MakeString(
        "num_buckets must contain only one element. It has ",
        num_buckets_dimensions.size(), " dimensions."));

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
      throw std::runtime_error(MakeString("Unexpected input index ", index));
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
    throw std::runtime_error(MakeString(
        "num_buckets must contain only one element. It has ",
        num_buckets_dimensions.size(), " dimensions."));

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
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpStringHashFast::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpStringHashFast::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"
#include "sentencepiece_tokenizer.hpp"
#include "kernels/string_common.h"
#include "base64.h"

KernelSentencepieceTokenizer::KernelSentencepieceTokenizer(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api) {
  std::string model_as_string = ort_.KernelInfoGetAttribute<std::string>(info, "model");
  sentencepiece::ModelProto model_proto;
  std::vector<uint8_t> model_as_bytes;
  if (base64_decode(model_as_string, model_as_bytes)) {
    model_proto.ParseFromArray(model_as_bytes.data(), model_as_bytes.size());
  } else {
    model_proto.ParseFromArray(model_as_string.c_str(), model_as_string.size());
  }
  sentencepiece::util::Status status = tokenizer_.Load(model_proto);
  if (!status.ok())
    throw std::runtime_error(MakeString(
        "Failed to create SentencePieceProcessor instance. Error code is ",
        (int)status.code(), ". Message is '", status.error_message(), "'."));
}

static void _check_dimension_constant(Ort::CustomOpApi ort, const OrtValue* ort_value, const char* name) {
  OrtTensorDimensions dimensions(ort, ort_value);
  if (dimensions.size() != 1 || dimensions[0] != 1)
    throw std::runtime_error(MakeString(
        name, " must contain only one element. It has ", dimensions.size(), " dimensions."));
}

void KernelSentencepieceTokenizer::Compute(OrtKernelContext* context) {
  // Update with the new API
  const OrtValue* ort_input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> str_input;
  GetTensorMutableDataString(ort_, context, ort_input, str_input);
  const OrtValue* ort_nbest_size = ort_.KernelContext_GetInput(context, 1);
  const float* p_nbest_size = ort_.GetTensorData<float>(ort_nbest_size);
  const OrtValue* ort_alpha = ort_.KernelContext_GetInput(context, 2);
  const float* p_alpha = ort_.GetTensorData<float>(ort_alpha);
  const OrtValue* ort_add_bos = ort_.KernelContext_GetInput(context, 3);
  const bool* p_add_bos = ort_.GetTensorData<bool>(ort_add_bos);
  const OrtValue* ort_add_eos = ort_.KernelContext_GetInput(context, 4);
  const bool* p_add_eos = ort_.GetTensorData<bool>(ort_add_eos);
  const OrtValue* ort_add_rev = ort_.KernelContext_GetInput(context, 5);
  const bool* p_add_rev = ort_.GetTensorData<bool>(ort_add_rev);

  // Verifications
  _check_dimension_constant(ort_, ort_nbest_size, "nbest_size");
  _check_dimension_constant(ort_, ort_alpha, "alpha");
  _check_dimension_constant(ort_, ort_add_bos, "add_bos");
  _check_dimension_constant(ort_, ort_add_eos, "add_eos");
  _check_dimension_constant(ort_, ort_add_rev, "add_rev");

  // computation

  std::vector<int64_t> indices;
  std::vector<int> content;
  indices.reserve(str_input.size() + 1);
  std::vector<int> inloop;
  for (size_t i = 0; i < str_input.size(); ++i) {
    if (!tokenizer_.Encode(str_input[i].c_str(), &inloop).ok())
      throw std::runtime_error(MakeString(
          "Unable to encode string '", str_input[i], "'."));
    indices.push_back(content.size());

    if (*p_add_rev) {
      if (*p_add_eos) {
        content.push_back(tokenizer_.eos_id());
      }
      content.insert(content.end(), inloop.rbegin(), inloop.rend());
      if (*p_add_bos) {
        content.push_back(tokenizer_.bos_id());
      }
    } else {
      if (*p_add_bos) {
        content.push_back(tokenizer_.bos_id());
      }
      content.insert(content.end(), inloop.begin(), inloop.end());
      if (*p_add_eos) {
        content.push_back(tokenizer_.eos_id());
      }
    }
  }
  indices.push_back(content.size());

  // Setup output
  std::vector<int64_t> size_content(1);
  size_content[0] = content.size();
  OrtValue* out_content = ort_.KernelContext_GetOutput(context, 0, size_content.data(), size_content.size());

  std::vector<int64_t> size_indices(1);
  size_indices[0] = indices.size();
  OrtValue* out_indices = ort_.KernelContext_GetOutput(context, 1, size_indices.data(), size_indices.size());

  int* ptr_content = ort_.GetTensorMutableData<int>(out_content);
  memcpy(ptr_content, content.data(), content.size() * sizeof(int));
  int64_t* ptr_indices = ort_.GetTensorMutableData<int64_t>(out_indices);
  memcpy(ptr_indices, indices.data(), indices.size() * sizeof(int64_t));
}

void* CustomOpSentencepieceTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelSentencepieceTokenizer(api, info);
};

const char* CustomOpSentencepieceTokenizer::GetName() const {
  return "SentencepieceTokenizer";
};

size_t CustomOpSentencepieceTokenizer::GetInputTypeCount() const {
  return 6;
};

ONNXTensorElementDataType CustomOpSentencepieceTokenizer::GetInputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 1:
    case 2:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case 3:
    case 4:
    case 5:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    default:
      throw std::runtime_error(MakeString("Unexpected input index ", index));
  }
};

size_t CustomOpSentencepieceTokenizer::GetOutputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpSentencepieceTokenizer::GetOutputType(size_t index) const {
  switch (index) {
    case 0:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
      throw std::runtime_error(MakeString("Unexpected output index ", index));
  }
};

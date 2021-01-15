// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sentencepiece_processor.h"
#include "sentencepiece_tokenizer.hpp"

KernelSentencepieceTokenizer::KernelSentencepieceTokenizer(OrtApi api) : BaseKernel(api) {
}

static void _check_dimension_constant(Ort::CustomOpApi ort, const OrtValue* ort_value, const char* name) {
  OrtTensorDimensions dimensions(ort, ort_value);
  if (dimensions.size() != 1 || dimensions[0] != 1)
    throw std::runtime_error(MakeString(
        name, " must contain only one element. It has ", dimensions.size(), " dimensions."));
}

void KernelSentencepieceTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* ort_model = ort_.KernelContext_GetInput(context, 0);
  const std::string* str_model = ort_.GetTensorData<std::string>(ort_model);
  const OrtValue* ort_input = ort_.KernelContext_GetInput(context, 1);
  const std::string* str_input = ort_.GetTensorData<std::string>(ort_input);
  const OrtValue* ort_nbest_size = ort_.KernelContext_GetInput(context, 2);
  const float* p_nbest_size = ort_.GetTensorData<float>(ort_nbest_size);
  const OrtValue* ort_alpha = ort_.KernelContext_GetInput(context, 3);
  const float* p_alpha = ort_.GetTensorData<float>(ort_alpha);
  const OrtValue* ort_add_bos = ort_.KernelContext_GetInput(context, 4);
  const bool* p_add_bos = ort_.GetTensorData<bool>(ort_add_bos);
  const OrtValue* ort_add_eos = ort_.KernelContext_GetInput(context, 5);
  const bool* p_add_eos = ort_.GetTensorData<bool>(ort_add_eos);
  const OrtValue* ort_add_rev = ort_.KernelContext_GetInput(context, 6);
  const bool* p_add_rev = ort_.GetTensorData<bool>(ort_add_rev);

  // Verifications
  _check_dimension_constant(ort_, ort_model, "model");
  _check_dimension_constant(ort_, ort_nbest_size, "nbest_size");
  _check_dimension_constant(ort_, ort_alpha, "alpha");
  _check_dimension_constant(ort_, ort_add_bos, "add_bos");
  _check_dimension_constant(ort_, ort_add_eos, "add_eos");
  _check_dimension_constant(ort_, ort_add_rev, "add_rev");

  // computation
  sentencepiece::SentencePieceProcessor* tokenizer = new sentencepiece::SentencePieceProcessor();
  sentencepiece::util::Status status = tokenizer->LoadFromSerializedProto(str_model->c_str());
  if (!status.ok())
    throw std::runtime_error("Failed to create SentencePieceProcessor instance.");

  OrtTensorDimensions dimensions(ort_, ort_input);
  std::vector<std::vector<int>> output(dimensions[0]);
  for (std::vector<std::vector<int>>::iterator it = output.begin(); it != output.end(); ++it, ++str_input) {
    if (!tokenizer->Encode(str_input->c_str(), &(*it)).ok())
      throw std::runtime_error(MakeString(
          "Unable to encode string '", *str_input, "'."));
  }

  delete tokenizer;

  // Setup output
  // OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
  // int64_t* out = ort_.GetTensorMutableData<int64_t>(output);
  // OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
  // int64_t size = ort_.GetTensorShapeElementCount(output_info);
  // ort_.ReleaseTensorTypeAndShapeInfo(output_info);

  throw std::runtime_error("not implemented yet");
}

void* CustomOpSentencepieceTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) const {
  return new KernelSentencepieceTokenizer(api);
};

const char* CustomOpSentencepieceTokenizer::GetName() const {
  return "SentencepieceTokenizer";
};

size_t CustomOpSentencepieceTokenizer::GetInputTypeCount() const {
  return 7;
};

ONNXTensorElementDataType CustomOpSentencepieceTokenizer::GetInputType(size_t index) const {
  switch (index) {
    case 0:
    case 1:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
    case 2:
    case 3:
      return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case 4:
    case 5:
    case 6:
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

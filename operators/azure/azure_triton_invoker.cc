// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "azure_triton_invoker.hpp"

////////////////////// AzureTritonInvoker //////////////////////

namespace tc = triton::client;

namespace ort_extensions {

AzureTritonInvoker::AzureTritonInvoker(const OrtApi& api, const OrtKernelInfo& info)
    : CloudBaseKernel(api, info) {
  auto err = tc::InferenceServerHttpClient::Create(&triton_client_, ModelUri(), Verbose());
}

std::string MapDataType(ONNXTensorElementDataType onnx_data_type) {
  std::string triton_data_type;
  switch (onnx_data_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      triton_data_type = "FP32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      triton_data_type = "UINT8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      triton_data_type = "INT8";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      triton_data_type = "UINT16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      triton_data_type = "INT16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      triton_data_type = "INT32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      triton_data_type = "INT64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
      triton_data_type = "BYTES";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
      triton_data_type = "BOOL";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      triton_data_type = "FP16";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      triton_data_type = "FP64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
      triton_data_type = "UINT32";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
      triton_data_type = "UINT64";
      break;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      triton_data_type = "BF16";
      break;
    default:
      break;
  }
  return triton_data_type;
}

int8_t* CreateNonStrTensor(const std::string& data_type,
                           ortc::Variadic& outputs,
                           size_t i,
                           const std::vector<int64_t>& shape) {
  if (data_type == "FP32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<float>(i, shape));
  } else if (data_type == "UINT8") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint8_t>(i, shape));
  } else if (data_type == "INT8") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int8_t>(i, shape));
  } else if (data_type == "UINT16") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint16_t>(i, shape));
  } else if (data_type == "INT16") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int16_t>(i, shape));
  } else if (data_type == "INT32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int32_t>(i, shape));
  } else if (data_type == "UINT32") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint32_t>(i, shape));
  } else if (data_type == "INT64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<int64_t>(i, shape));
  } else if (data_type == "UINT64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<uint64_t>(i, shape));
  } else if (data_type == "BOOL") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<bool>(i, shape));
  } else if (data_type == "FP64") {
    return reinterpret_cast<int8_t*>(outputs.AllocateOutput<double>(i, shape));
  } else {
    return {};
  }
}

#define CHECK_TRITON_ERR(ret, msg)                                                    \
  if (!ret.IsOk()) {                                                                  \
    ORTX_CXX_API_THROW("Triton err: " + ret.Message(), ORT_RUNTIME_EXCEPTION); \
  }

void AzureTritonInvoker::Compute(const ortc::Variadic& inputs, ortc::Variadic& outputs) const {
  auto auth_token = GetAuthToken(inputs);

  gsl::span<const std::string> input_names = InputNames();
  if (inputs.Size() != input_names.size()) {
    ORTX_CXX_API_THROW("input count mismatch", ORT_RUNTIME_EXCEPTION);
  }

  std::vector<std::unique_ptr<tc::InferInput>> triton_input_vec;
  std::vector<tc::InferInput*> triton_inputs;
  std::vector<std::unique_ptr<const tc::InferRequestedOutput>> triton_output_vec;
  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  tc::Error err;

  const auto& property_names = RequestPropertyNames();

  for (size_t ith_input = 1; ith_input < inputs.Size(); ++ith_input) {
    tc::InferInput* triton_input = {};
    std::string triton_data_type = MapDataType(inputs[ith_input]->Type());
    if (triton_data_type.empty()) {
      ORTX_CXX_API_THROW("unknow onnx data type", ORT_RUNTIME_EXCEPTION);
    }

    err = tc::InferInput::Create(&triton_input, property_names[ith_input], inputs[ith_input]->Shape(),
                                 triton_data_type);
    CHECK_TRITON_ERR(err, "failed to create triton input");
    triton_input_vec.emplace_back(triton_input);

    triton_inputs.push_back(triton_input);
    if ("BYTES" == triton_data_type) {
      const auto* string_tensor = reinterpret_cast<const ortc::Tensor<std::string>*>(inputs[ith_input].get());
      triton_input->AppendFromString(string_tensor->Data());
    } else {
      const float* data_raw = reinterpret_cast<const float*>(inputs[ith_input]->DataRaw());
      size_t size_in_bytes = inputs[ith_input]->SizeInBytes();
      err = triton_input->AppendRaw(reinterpret_cast<const uint8_t*>(data_raw), size_in_bytes);
      CHECK_TRITON_ERR(err, "failed to append raw data to input");
    }
  }

  gsl::span<const std::string> output_names = OutputNames();
  for (size_t ith_output = 0; ith_output < output_names.size(); ++ith_output) {
    tc::InferRequestedOutput* triton_output = {};
    err = tc::InferRequestedOutput::Create(&triton_output, output_names[ith_output]);
    CHECK_TRITON_ERR(err, "failed to create triton output");
    triton_output_vec.emplace_back(triton_output);
    triton_outputs.push_back(triton_output);
  }

  std::unique_ptr<tc::InferResult> results_ptr;
  tc::InferResult* results = {};
  tc::InferOptions options(ModelName());
  options.model_version_ = ModelVersion();
  options.client_timeout_ = 0;

  tc::Headers http_headers;
  http_headers["Authorization"] = std::string{"Bearer "} + auth_token;

  err = triton_client_->Infer(&results, options, triton_inputs, triton_outputs,
                              http_headers, tc::Parameters(),
                              tc::InferenceServerHttpClient::CompressionType::NONE,  // support compression in config?
                              tc::InferenceServerHttpClient::CompressionType::NONE);

  results_ptr.reset(results);
  CHECK_TRITON_ERR(err, "failed to do triton inference");

  size_t output_index = 0;

  for (const auto& output_name : output_names) {
    std::vector<int64_t> shape;
    err = results_ptr->Shape(output_name, &shape);
    CHECK_TRITON_ERR(err, "failed to get output shape");

    std::string type;
    err = results_ptr->Datatype(output_name, &type);
    CHECK_TRITON_ERR(err, "failed to get output type");

    if ("BYTES" == type) {
      std::vector<std::string> output_strings;
      err = results_ptr->StringData(output_name, &output_strings);
      CHECK_TRITON_ERR(err, "failed to get output as string");
      auto& string_tensor = outputs.AllocateStringTensor(output_index);
      string_tensor.SetStringOutput(output_strings, shape);
    } else {
      const uint8_t* raw_data = {};
      size_t raw_size;
      err = results_ptr->RawData(output_name, &raw_data, &raw_size);
      CHECK_TRITON_ERR(err, "failed to get output raw data");
      auto* output_raw = CreateNonStrTensor(type, outputs, output_index, shape);
      memcpy(output_raw, raw_data, raw_size);
    }

    ++output_index;
  }
}

}  // namespace ort_extensions

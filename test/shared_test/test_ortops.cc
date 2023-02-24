// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <filesystem>
#include "gtest/gtest.h"
#include "ocos.h"
#include "ustring.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "test_kernel.hpp"

const char* GetLibraryPath() {
#if defined(_WIN32)
  return "ortextensions.dll";
#elif defined(__APPLE__)
  return "libortextensions.dylib";
#elif defined(ANDROID) || defined(__ANDROID__)
  return "libortextensions.so";
#else
  return "lib/libortextensions.so";
#endif
}

struct KernelOne : BaseKernel {
  KernelOne(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const OrtValue* input_Y = ort_.KernelContext_GetInput(context, 1);
    const float* X = ort_.GetTensorData<float>(input_X);
    const float* Y = ort_.GetTensorData<float>(input_Y);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    float* out = ort_.GetTensorMutableData<float>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i] + Y[i];
    }
  }
};

struct CustomOpOne : OrtW::CustomOpBase<CustomOpOne, KernelOne> {
  const char* GetName() const {
    return "CustomOpOne";
  };
  size_t GetInputTypeCount() const {
    return 2;
  };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };
  size_t GetOutputTypeCount() const {
    return 1;
  };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };
};

struct KernelTwo : BaseKernel {
  KernelTwo(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const float* X = ort_.GetTensorData<float>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);

    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int32_t* out = ort_.GetTensorMutableData<int32_t>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = (int32_t)(round(X[i]));
    }
  }
};

struct CustomOpTwo : OrtW::CustomOpBase<CustomOpTwo, KernelTwo> {
  const char* GetName() const {
    return "CustomOpTwo";
  };
  size_t GetInputTypeCount() const {
    return 1;
  };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  };
  size_t GetOutputTypeCount() const {
    return 1;
  };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  };
};

struct KernelThree : BaseKernel {
  KernelThree(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    if (!TryToGetAttribute("substr", substr_)) {
      substr_ = "";
    }
  }
  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_val = ort_.KernelContext_GetInput(context, 0);
    std::vector<std::string> input_strs;
    GetTensorMutableDataString(api_, ort_, context, input_val, input_strs);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_val);
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    int64_t* out = ort_.GetTensorMutableData<int64_t>(output);

    // Record substring locations in output
    for (int64_t i = 0; i < dimensions.Size(); i++) {
      out[i] = input_strs[i].find(substr_);
    }
  }

 private:
  std::string substr_;
};

struct CustomOpThree : OrtW::CustomOpBase<CustomOpThree, KernelThree> {
  // This is  example code to show how to override the CustomOpBase::CreateKernel method even though it is not virtual.
  // The CustomOpBase implementation will call the CreateKernel of the first class specified in the template,
  // and from there it's also possible to call the base CreateKernel as per below.
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo& info) const {
    std::cout << "Called CreateKernel override" << std::endl;
    return OrtW::CustomOpBase<CustomOpThree, KernelThree>::CreateKernel(api, info);
  };
  const char* GetName() const {
    return "CustomOpThree";
  };
  size_t GetInputTypeCount() const {
    return 1;
  };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  };
  size_t GetOutputTypeCount() const {
    return 1;
  };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  };
};

template <typename T>
void _emplace_back(Ort::MemoryInfo& memory_info, std::vector<Ort::Value>& ort_inputs, const std::vector<T>& values, const std::vector<int64_t>& dims) {
  ort_inputs.emplace_back(Ort::Value::CreateTensor<T>(
      memory_info, const_cast<T*>(values.data()), values.size(), dims.data(), dims.size()));
}

// std::vector<bool> doesn't have data(), so we need to covert it to uint8_t and reinterpret to bool
template <>
void _emplace_back(Ort::MemoryInfo& memory_info, std::vector<Ort::Value>& ort_inputs, const std::vector<bool>& values, const std::vector<int64_t>& dims) {
  static std::vector<uint8_t> convertor;
  convertor.resize(values.size());
  for (int i = 0; i < values.size(); i++) {
    convertor[i] = values[i];
  }

  ort_inputs.emplace_back(Ort::Value::CreateTensor<>(
      memory_info, reinterpret_cast<bool*>(convertor.data()), values.size(), dims.data(), dims.size()));
}

template <typename T>
void _assert_eq(Ort::Value& output_tensor, const std::vector<T>& expected, size_t total_len) {
  ASSERT_EQ(expected.size(), total_len);
  T* f = output_tensor.GetTensorMutableData<T>();
  for (size_t i = 0; i != total_len; ++i) {
    ASSERT_EQ(expected[i], f[i]);
  }
}

void GetTensorMutableDataString(const OrtApi& api, const OrtValue* value, std::vector<std::string>& output) {
  OrtTensorDimensions dimensions(OrtW::CustomOpApi(api), value);
  size_t len = static_cast<size_t>(dimensions.Size());
  size_t data_len;
  OrtW::ThrowOnError(api, api.GetStringTensorDataLength(value, &data_len));
  output.resize(len);
  std::vector<char> result(data_len + len + 1, '\0');
  std::vector<size_t> offsets(len);
  OrtW::ThrowOnError(api, api.GetStringTensorContent(value, (void*)result.data(), data_len, offsets.data(), offsets.size()));
  output.resize(len);
  for (int64_t i = (int64_t)len - 1; i >= 0; --i) {
    if (i < len - 1)
      result[offsets[i + (int64_t)1]] = '\0';
    output[i] = result.data() + offsets[i];
  }
}

void RunSession(Ort::Session& session_object,
                const std::vector<TestValue>& inputs,
                const std::vector<TestValue>& outputs) {
  std::vector<Ort::Value> ort_inputs;
  std::vector<const char*> input_names;
  std::vector<const char*> output_names;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  Ort::AllocatorWithDefaultOptions allocator;

  for (size_t i = 0; i < inputs.size(); i++) {
    input_names.emplace_back(inputs[i].name);
    switch (inputs[i].element_type) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
        _emplace_back(memory_info, ort_inputs, inputs[i].values_bool, inputs[i].dims);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        _emplace_back(memory_info, ort_inputs, inputs[i].values_float, inputs[i].dims);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        _emplace_back(memory_info, ort_inputs, inputs[i].values_uint8, inputs[i].dims);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        _emplace_back(memory_info, ort_inputs, inputs[i].values_int32, inputs[i].dims);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        _emplace_back(memory_info, ort_inputs, inputs[i].values_int64, inputs[i].dims);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
        Ort::Value& ort_value = ort_inputs.emplace_back(
            Ort::Value::CreateTensor(allocator, inputs[i].dims.data(), inputs[i].dims.size(),
                                     ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));
        for (size_t i_str = 0; i_str < inputs[i].values_string.size(); ++i_str) {
          ort_value.FillStringTensorElement(inputs[i].values_string[i_str].c_str(), i_str);
        }
      } break;
      default:
        throw std::runtime_error(MakeString(
            "Unable to handle input ", i, " type ", inputs[i].element_type,
            " is not implemented yet."));
    }
  }
  for (size_t index = 0; index < outputs.size(); ++index) {
    output_names.push_back(outputs[index].name);
  }

  std::vector<Ort::Value> ort_outputs;
  ort_outputs = session_object.Run(Ort::RunOptions{nullptr},
                                   input_names.data(), ort_inputs.data(), ort_inputs.size(),
                                   output_names.data(), outputs.size());
  ASSERT_EQ(outputs.size(), ort_outputs.size());
  for (size_t index = 0; index < outputs.size(); ++index) {
    auto output_tensor = &ort_outputs[index];
    const TestValue& expected = outputs[index];

    auto type_info = output_tensor->GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = type_info.GetElementType();
    ASSERT_EQ(output_type, expected.element_type);
    std::vector<int64_t> dimension = type_info.GetShape();
    ASSERT_EQ(dimension, expected.dims);
    size_t total_len = type_info.GetElementCount();
    switch (expected.element_type) {
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        _assert_eq(*output_tensor, expected.values_float, total_len);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        _assert_eq(*output_tensor, expected.values_uint8, total_len);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        _assert_eq(*output_tensor, expected.values_int32, total_len);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        _assert_eq(*output_tensor, expected.values_int64, total_len);
        break;
      case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
        std::vector<std::string> output_string;
        GetTensorMutableDataString(Ort::GetApi(), *output_tensor, output_string);
        ASSERT_EQ(expected.values_string, output_string);
        break;
      }
      default:
        throw std::runtime_error(MakeString(
            "Unable to handle output ", index, " type ", expected.element_type,
            " is not implemented yet."));
    }
  }
}

void TestInference(Ort::Env& env, const ORTCHAR_T* model_uri,
                   const std::vector<TestValue>& inputs,
                   const std::vector<TestValue>& outputs,
                   const char* custom_op_library_filename) {
  Ort::SessionOptions session_options;
  void* handle = nullptr;
  if (custom_op_library_filename) {
    Ort::ThrowOnError(Ort::GetApi().RegisterCustomOpsLibrary((OrtSessionOptions*)session_options,
                                                             custom_op_library_filename, &handle));
  }

  // if session creation passes, model loads fine
  Ort::Session session(env, model_uri, session_options);

  // Now run
  RunSession(session, inputs, outputs);
}

static CustomOpOne op_1st;
static CustomOpTwo op_2nd;

TEST(utils, test_ort_case) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  std::vector<TestValue> inputs(2);
  inputs[0].name = "input_1";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  inputs[0].dims = {3, 5};
  inputs[0].values_float = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f,
                            6.6f, 7.7f, 8.8f, 9.9f, 10.0f,
                            11.1f, 12.2f, 13.3f, 14.4f, 15.5f};
  inputs[1].name = "input_2";
  inputs[1].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
  inputs[1].dims = {3, 5};
  inputs[1].values_float = {15.5f, 14.4f, 13.3f, 12.2f, 11.1f,
                            10.0f, 9.9f, 8.8f, 7.7f, 6.6f,
                            5.5f, 4.4f, 3.3f, 2.2f, 1.1f};

  // prepare expected inputs and outputs
  std::vector<TestValue> outputs(1);
  outputs[0].name = "output";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
  outputs[0].dims = {3, 5};
  outputs[0].values_int32 = {17, 17, 17, 17, 17,
                             17, 18, 18, 18, 17,
                             17, 17, 17, 17, 17};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_test.onnx";
  AddExternalCustomOp(&op_1st);
  AddExternalCustomOp(&op_2nd);
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());
}

static CustomOpThree op_3rd;

TEST(utils, test_get_str_attr) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");

  // Input: list of strings
  std::vector<TestValue> inputs(1);
  inputs[0].name = "input_1";
  inputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  inputs[0].dims = {3};
  inputs[0].values_string = {"look for abc", "abc is first", "not found here"};

  // Expected output: location of the substring "abc" in each input string.
  std::vector<TestValue> outputs(1);
  outputs[0].name = "output_1";
  outputs[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs[0].dims = {3};
  outputs[0].values_int64 = {9, 0, -1};

  std::filesystem::path model_path = "data";
  model_path /= "custom_op_str_attr_test.onnx";
  AddExternalCustomOp(&op_3rd);
  TestInference(*ort_env, model_path.c_str(), inputs, outputs, GetLibraryPath());

  // Expected output when the attribute is missing from the node.
  std::vector<TestValue> outputs_missing(1);
  outputs_missing[0].name = "output_1";
  outputs_missing[0].element_type = ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  outputs_missing[0].dims = {3};
  outputs_missing[0].values_int64 = {0, 0, 0};

  std::filesystem::path model_missing_attr_path = "data";
  model_missing_attr_path /= "custom_op_str_attr_missing_test.onnx";
  TestInference(*ort_env, model_missing_attr_path.c_str(), inputs, outputs_missing, GetLibraryPath());
}

TEST(ustring, tensor_operator) {
  OrtValue* tensor;
  OrtAllocator* allocator;

  const auto* api_base = OrtGetApiBase();
  const auto* api = api_base->GetApi(ORT_API_VERSION);
  auto status = api->GetAllocatorWithDefaultOptions(&allocator);
  auto err_code = ORT_OK;
  if (status != nullptr) {
    err_code = api->GetErrorCode(status);
    api->ReleaseStatus(status);
  }
  EXPECT_EQ(err_code, ORT_OK);

  OrtW::CustomOpApi custom_api(*api);

  std::vector<int64_t> dim{2, 2};
  status = api->CreateTensorAsOrtValue(allocator, dim.data(), dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING, &tensor);
  if (status != nullptr) {
    err_code = api->GetErrorCode(status);
    api->ReleaseStatus(status);
  }
  EXPECT_EQ(err_code, ORT_OK);

  std::vector<ustring> input_value{ustring("test"), ustring("测试"), ustring("Test de"), ustring("🧐")};
  FillTensorDataString(*api, custom_api, nullptr, input_value, tensor);

  std::vector<ustring> output_value;
  GetTensorMutableDataString(*api, custom_api, nullptr, tensor, output_value);

  EXPECT_EQ(input_value, output_value);
}

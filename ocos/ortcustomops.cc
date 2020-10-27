// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "kernels/kernels.h"
#include "helpers/utils.h"

#include <vector>
#include <cmath>
#include <algorithm>

struct OrtTensorDimensions : std::vector<int64_t> {
  OrtTensorDimensions(Ort::CustomOpApi& ort, const OrtValue* value) {
    OrtTensorTypeAndShapeInfo* info = ort.GetTensorTypeAndShape(value);
    std::vector<int64_t>::operator=(ort.GetTensorShape(info));
    ort.ReleaseTensorTypeAndShapeInfo(info);
  }
};

struct KernelOne {
  KernelOne(OrtApi api)
      : api_(api),
        ort_(api_) {
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

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelTwo {
  KernelTwo(OrtApi api)
      : api_(api),
        ort_(api_) {
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

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelStringUpper {
  KernelStringUpper(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const std::string* X = ort_.GetTensorData<std::string>(input_X);

    // Setup output
    OrtTensorDimensions dimensions(ort_, input_X);
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), dimensions.size());
    std::string* out = ort_.GetTensorMutableData<std::string>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    for (int64_t i = 0; i < size; i++) {
      out[i] = X[i];
      std::transform(out[i].begin(), out[i].end(), out[i].begin(), ::toupper);
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct KernelStringJoin {
  KernelStringJoin(OrtApi api)
      : api_(api),
        ort_(api_) {
  }

  void Compute(OrtKernelContext* context) {
    // Setup inputs
    const OrtValue* input_X = ort_.KernelContext_GetInput(context, 0);
    const std::string* X = ort_.GetTensorData<std::string>(input_X);
    const OrtValue* input_sep = ort_.KernelContext_GetInput(context, 1);
    const std::string* sep = ort_.GetTensorData<std::string>(input_sep);

    // Setup output
    OrtTensorDimensions dimensions_sep(ort_, input_sep);
    if (dimensions_sep.size() != 1 || dimensions_sep[0] != 1)
      throw std::runtime_error("Input 2 is the separator, it has 1 element.");
    OrtTensorDimensions dimensions(ort_, input_X);
    if (dimensions.size() != 2)
      throw std::runtime_error(MakeString("Input 1 must have 2 dimensions but has ", dimensions.size(), "."));
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, dimensions.data(), 1);
    std::string* out = ort_.GetTensorMutableData<std::string>(output);

    OrtTensorTypeAndShapeInfo* output_info = ort_.GetTensorTypeAndShape(output);
    int64_t size = ort_.GetTensorShapeElementCount(output_info);
    ort_.ReleaseTensorTypeAndShapeInfo(output_info);

    // Do computation
    int64_t index = 0;
    for (int64_t i = 0; i < size; ++i) {
      std::ostringstream st;
      for (int64_t j = 0; j < dimensions[1] - 1; ++j, ++index) {
        st << X[index] << *sep;
      }
      st << X[index++];
      out[i] = st.str();
    }
  }

 private:
  OrtApi api_;  // keep a copy of the struct, whose ref is used in the ort_
  Ort::CustomOpApi ort_;
};

struct CustomOpOne : Ort::CustomOpBase<CustomOpOne, KernelOne> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelOne(api);
  };

  const char* GetName() const { return "CustomOpOne"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

} c_CustomOpOne;

struct CustomOpTwo : Ort::CustomOpBase<CustomOpTwo, KernelTwo> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelTwo(api);
  };

  const char* GetName() const { return "CustomOpTwo"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32; };

} c_CustomOpTwo;

struct CustomOpStringUpper : Ort::CustomOpBase<CustomOpStringUpper, KernelStringUpper> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelStringUpper(api);
  };

  const char* GetName() const { return "StringUpper"; };

  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; };

} c_CustomOpStringUpper;

struct CustomOpStringJoin : Ort::CustomOpBase<CustomOpStringJoin, KernelStringJoin> {
  void* CreateKernel(OrtApi api, const OrtKernelInfo* /* info */) {
    return new KernelStringJoin(api);
  };

  const char* GetName() const { return "StringJoin"; };

  size_t GetInputTypeCount() const { return 2; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; };

  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING; };

} c_CustomOpStringJoin;

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain("test.customop", &domain)) {
    return status;
  } else {
    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpOne)) {
      return status;
    }

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpTwo)) {
      return status;
    }

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpStringUpper)) {
      return status;
    }

    if (auto status = ortApi->CustomOpDomain_Add(domain, &c_CustomOpStringJoin)) {
      return status;
    }

    if (auto status = ortApi->AddCustomOpDomain(options, domain)) {
      return status;
    }
  }

  domain = nullptr;
  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  auto custom_op_list = GetCustomOpSchemaList();
  for (auto it = custom_op_list; *it != nullptr; ++it) {
    auto obj_ptr = (*it)();
    // TODO: it doesn't make sense ORT needs non-const OrtCustomOp object, will fix in new ORT release
    OrtCustomOp* op_ptr = const_cast<OrtCustomOp*>(obj_ptr);
    if (auto status = ortApi->CustomOpDomain_Add(domain, op_ptr)) {
      return status;
    }
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    OrtCustomOp* op_ptr = const_cast<OrtCustomOp*>(c_ops);
    auto status = ortApi->CustomOpDomain_Add(domain, op_ptr);
    if (status) {
      return status;
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

  return ortApi->AddCustomOpDomain(options, domain);
}

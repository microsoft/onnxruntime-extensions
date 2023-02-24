// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <filesystem>

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "ocos.h"
#include "test_kernel.hpp"

// throw in ctor which will be called during model load
struct ExceptionalKernel1 : BaseKernel {
  ExceptionalKernel1(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    ORTX_CXX_API_THROW("Throw in ctor", ORT_FAIL);
  }

  void Compute(OrtKernelContext* context) {}
};

// throw in Compute which will be called during model execution
struct ExceptionalKernel2 : BaseKernel {
  ExceptionalKernel2(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  }

  void Compute(OrtKernelContext* context) {
    ORTX_CXX_API_THROW("Throw in Compute", ORT_FAIL);
  }
};

struct ExceptionalCustomOp1 : OrtW::CustomOpBase<ExceptionalCustomOp1, ExceptionalKernel1> {
  const char* GetName() const { return "ExceptionalCustomOp1"; };
  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

struct ExceptionalCustomOp2 : OrtW::CustomOpBase<ExceptionalCustomOp2, ExceptionalKernel2> {
  const char* GetName() const { return "ExceptionalCustomOp2"; };
  size_t GetInputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetInputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
  size_t GetOutputTypeCount() const { return 1; };
  ONNXTensorElementDataType GetOutputType(size_t /*index*/) const { return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT; };
};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);
static ExceptionalCustomOp1 custom_op1;
static ExceptionalCustomOp2 custom_op2;

// test a call to an entry point wrapped with OCOS_API_IMPL_BEGIN/OCOS_API_IMPL_END behaves as expected.
// the throw in the ctor of ExceptionalCustomOp1 should be triggered during model loading.
TEST(Exceptions, TestApiTryCatch_ThrowInModelLoad) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  AddExternalCustomOp(&custom_op1);

  Ort::SessionOptions session_options;
  RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase());

  std::filesystem::path model("data/exceptional_custom_op1.onnx");
  auto fail_fn = [&]() {
    Ort::Session session(*ort_env, model.c_str(), session_options);
  };

// if no exceptions, the ORTX_CXX_API_THROW will trigger the log+abort
// if no exception propagation, the OCOS_API_IMPL_END will trigger the log+abort
#if defined(OCOS_NO_EXCEPTIONS) || defined(OCOS_PREVENT_EXCEPTION_PROPAGATION)
  // the exception should be caught and logged, and the process should abort so the exception is not propagated up.
  // log output needs to be manually checked
  // can test on Linux but not Windows.
#if !defined(_WIN32)
  EXPECT_EXIT(fail_fn(), ::testing::KilledBySignal(SIGABRT), ".*");
#endif
#else
  // ORT catches the exceptions thrown by the custom op and rethrows them as Ort::Exception
  EXPECT_THROW(fail_fn(), Ort::Exception);
#endif
}

// test a call to an entry point wrapped with OCOS_API_IMPL_BEGIN/OCOS_API_IMPL_END behaves as expected.
// the throw in the Compute of ExceptionalCustomOp2 should be triggered during model execution.
TEST(Exceptions, TestApiTryCatch_ThrowInModelExecution) {
  auto ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Default");
  AddExternalCustomOp(&custom_op2);

  Ort::SessionOptions session_options;
  RegisterCustomOps((OrtSessionOptions*)session_options, OrtGetApiBase());

  std::filesystem::path model("data/exceptional_custom_op2.onnx");
  Ort::Session session(*ort_env, model.c_str(), session_options);
  Ort::AllocatorWithDefaultOptions allocator;

  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const char* input_names[] = {"A"};
  const char* output_names[] = {"B"};

  std::vector<int64_t> dims = {2};
  std::vector<float> input = {0.f, 1.f};
  std::vector<Ort::Value> ort_input;
  ort_input.push_back(Ort::Value::CreateTensor<float>(memory_info, input.data(), input.size(),
                                                      dims.data(), dims.size()));

  auto fail_fn = [&]() {
    // executing the model should call Compute of the custom op, which should throw
    std::vector<Ort::Value> ort_outputs;
    ort_outputs = session.Run(Ort::RunOptions{nullptr}, input_names, ort_input.data(), ort_input.size(),
                              output_names, 1);
  };

// if no exceptions, the ORTX_CXX_API_THROW will trigger the log+abort
// if no exception propagation, the OCOS_API_IMPL_END will trigger the log+abort
#if defined(OCOS_NO_EXCEPTIONS) || defined(OCOS_PREVENT_EXCEPTION_PROPAGATION)
  // can test on Linux but not Windows
#if !defined(_WIN32)
  EXPECT_EXIT(fail_fn(), ::testing::KilledBySignal(SIGABRT), ".*");
#endif
#else
  // ORT catches the exceptions thrown by the custom op and rethrows them as Ort::Exception
  EXPECT_THROW(fail_fn(), Ort::Exception);
#endif
}

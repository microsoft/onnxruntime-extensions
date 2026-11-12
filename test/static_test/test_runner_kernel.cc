// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdint>
#include <vector>
#include "gtest/gtest.h"
#include "c_api_utils.hpp"
#include "runner.hpp"

using namespace ort_extensions;

static OrtxStatus CopyFloats(const ortc::Tensor<float>& in, ortc::Tensor<float>& out) {
  auto* data = out.Allocate(in.Shape());
  std::copy(in.Data(), in.Data() + in.NumberOfElement(), data);
  return {};
}

TEST(KernelRunnerTest, InvokeRejectsMismatchedInputTensorType) {
  auto kernel = CreateKernelInstance(&CopyFloats);
  auto outputs = kernel->AllocateOutput(&CppAllocator::Instance());

  std::vector<float> floats{1.f, 2.f};
  std::vector<int64_t> ints{1, 2};
  ortc::Tensor<float> float_input({2}, floats.data());
  ortc::Tensor<int64_t> int_input({2}, ints.data());

  TensorArgs bad_inputs{&int_input};
  EXPECT_FALSE(kernel->Invoke(bad_inputs, outputs).IsOk());

  TensorArgs no_inputs;
  EXPECT_FALSE(kernel->Invoke(no_inputs, outputs).IsOk());

  TensorArgs good_inputs{&float_input};
  EXPECT_TRUE(kernel->Invoke(good_inputs, outputs).IsOk());

  for (auto* t : outputs) {
    delete t;
  }
}
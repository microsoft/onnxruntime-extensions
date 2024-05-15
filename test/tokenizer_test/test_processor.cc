// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"
#include "shared/api/image_processor.h"

using namespace ort_extensions;

TEST(MultiModelTest, TestPhi_3) {

  auto [input_data, n_data] = ort_extensions::LoadRawImages(
      {"data/processor/australia.jpg", "data/processor/exceltable.png"});

  auto proc = OrtxObjectPtr<ImageProcessor>(OrtxCreateProcessor, "data/processor/image_processor.json");
  ortc::Tensor<float>* pixel_values;
  ortc::Tensor<int64_t>* image_sizes;
  ortc::Tensor<int64_t>* num_img_takens;

  auto [status, r] = proc->PreProcess(
      ort_extensions::span(input_data.get(), (size_t)n_data),
      &pixel_values,
      &image_sizes,
      &num_img_takens);

  ASSERT_TRUE(status.IsOk());
  ASSERT_EQ(pixel_values->Shape(), std::vector<int64_t>({2, 17, 3, 336, 336}));

  proc->ClearOutputs(&r);
}

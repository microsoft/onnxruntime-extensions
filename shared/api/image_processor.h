// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <tuple>
#include <vector>
#include <unordered_map>

#include "ortx_processor.h"
#include "c_api_utils.hpp"

#include "runner.hpp"

namespace ort_extensions {

using ImageRawData = std::vector<uint8_t>;

std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(
  const std::initializer_list<const char*>& image_paths);

std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(
  const char* image_paths[], size_t num_images);

class ProcessorResult : public OrtxObjectImpl {
 public:
  ProcessorResult() : OrtxObjectImpl(kOrtxKindProcessorResult) {}
  ortc::Tensor<float>* pixel_values{};
  ortc::Tensor<int64_t>* image_sizes{};
  ortc::Tensor<int64_t>* num_img_tokens{};
};

class ImageProcessor : public OrtxObjectImpl {
 public:
  ImageProcessor();
  virtual ~ImageProcessor() = default;

  OrtxStatus Init(std::string_view processor_def);

  // Deprecated, using the next function instead
  std::tuple<OrtxStatus, ProcessorResult> PreProcess(ort_extensions::span<ImageRawData> image_data,
                                                     ortc::Tensor<float>** pixel_values,
                                                     ortc::Tensor<int64_t>** image_sizes,
                                                     ortc::Tensor<int64_t>** num_img_takens) const;

  OrtxStatus PreProcess(ort_extensions::span<ImageRawData> image_data, TensorResult& r) const;

  // Deprecated, using the next function instead
  static void ClearOutputs(ProcessorResult* r);

  static Operation::KernelRegistry kernel_registry_;

 private:
  ExecutionPlan op_plan_;
  std::string   name_;
  std::string   output_aligner_;
};

}  // namespace ort_extensions

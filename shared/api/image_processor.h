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

template <typename It>
std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(It begin, It end);

std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(
    const std::initializer_list<const char*>& image_paths);

class ProcessorResult : public OrtxObjectImpl {
 public:
  ProcessorResult() : OrtxObjectImpl(kOrtxKindProcessorResult) {}
  ortc::Tensor<float>* pixel_values{};
  ortc::Tensor<int64_t>* image_sizes{};
  ortc::Tensor<int64_t>* num_img_takens{};
};

class ImageProcessorResult : public OrtxObjectImpl {
 public:
  ImageProcessorResult() : OrtxObjectImpl(kOrtxKindImageProcessorResult) {}
  std::vector<TensorPtr> results;
};

class ImageProcessor : public OrtxObjectImpl {
 public:
  ImageProcessor();
  virtual ~ImageProcessor() = default;

  OrtxStatus Init(std::string_view processor_def);

  std::tuple<OrtxStatus, ProcessorResult> PreProcess(ort_extensions::span<ImageRawData> image_data,
                                                     ortc::Tensor<float>** pixel_values,
                                                     ortc::Tensor<int64_t>** image_sizes,
                                                     ortc::Tensor<int64_t>** num_img_takens) const;

  OrtxStatus PreProcess(ort_extensions::span<ImageRawData> image_data, ImageProcessorResult& r) const;

  static void ClearOutputs(ProcessorResult* r);
  static void ClearOutputs(ImageProcessorResult* r);

  static Operation::KernelRegistry kernel_registry_;

 private:
  std::vector<std::unique_ptr<Operation>> operations_;
  ortc::IAllocator* allocator_;
};

}  // namespace ort_extensions

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "nlohmann/json.hpp"
#include "file_sys.h"

#include "image_processor.h"
#include "c_api_utils.hpp"
#if OCOS_ENABLE_VENDOR_IMAGE_CODECS
  #if WIN32
    #include "image_decoder_win32.hpp"
  #elif __APPLE__
    #include "image_decoder_darwin.hpp"
  #else
    #include "image_decoder.hpp"
  #endif
#else
  #include "image_decoder.hpp"
#endif

#include "image_transforms.hpp"
#include "image_transforms_phi_3.hpp"
#include "image_transforms_mllama.hpp"

namespace ort_extensions {
std::tuple<std::unique_ptr<ImageRawData[]>, size_t>
LoadRawImages(const std::initializer_list<const char*>& image_paths) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths.begin(), image_paths.end());
}

std::tuple<std::unique_ptr<ImageRawData[]>, size_t>
LoadRawImages(const char* image_paths[], size_t num_images) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths, image_paths + num_images);
}
}  // namespace ort_extensions

using namespace ort_extensions;
using json = nlohmann::json;

Operation::KernelRegistry ImageProcessor::kernel_registry_ = {
    {"DecodeImage", []() { return CreateKernelInstance(&DecodeImage::Compute); }},
    {"Resize", []() { return CreateKernelInstance(&Resize::Compute); }},
    {"Rescale", []() { return CreateKernelInstance(&Rescale::Compute); }},
    {"Normalize", []() { return CreateKernelInstance(&Normalize::Compute); }},
    {"CenterCrop", []() { return CreateKernelInstance(&CenterCrop::Compute); }},
    {"ConvertRGB", []() { return CreateKernelInstance(convert_to_rgb); }},
    {"Permute3D", []() { return CreateKernelInstance(&Permute3D::Compute); }},
    {"Phi3ImageTransform", []() { return CreateKernelInstance(phi3_hd_transform); }},
    {"Llama3ImageTransform", []() { return CreateKernelInstance(&Llama3ImageTransform::Compute); }},
};

OrtxStatus ImageProcessor::Init(std::string_view processor_def) {
  std::string processor_def_str;
  if (processor_def.size() >= 5 && processor_def.substr(processor_def.size() - 5) == ".json") {
    std::ifstream ifs = path({processor_def.data(), processor_def.size()}).open();
    if (!ifs.is_open()) {
      return {kOrtxErrorInvalidArgument, std::string("[ImageProcessor]: failed to open ") + std::string(processor_def)};
    }

    processor_def_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    processor_def = processor_def_str.c_str();
  }
  // pase the processor_def by json
  auto proc_json = json::parse(processor_def, nullptr, false);
  if (proc_json.is_discarded()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: failed to parse processor json configuration."};
  }

  auto processor_root = proc_json.at("processor");
  if (!processor_root.is_object()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: processor field is missing."};
  }

  auto transforms = processor_root.at("transforms");
  if (!transforms.is_array() || transforms.empty()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: transforms field is missing."};
  }

  operations_.reserve(transforms.size());
  for (auto mod_iter = transforms.begin(); mod_iter != transforms.end(); ++mod_iter) {
    auto op = std::make_unique<Operation>(kernel_registry_);
    auto status = op->Init(mod_iter->dump());
    if (!status.IsOk()) {
      return status;
    }

    operations_.push_back(std::move(op));
  }

  return {};
}

ImageProcessor::ImageProcessor() : OrtxObjectImpl(kOrtxKindProcessor), allocator_(&CppAllocator::Instance()) {}

template <typename T>
static ortc::Tensor<T>* StackTensor(const std::vector<TensorArgs>& arg_lists, int axis, ortc::IAllocator* allocator) {
  using TT = ortc::Tensor<T>;
  auto output = std::make_unique<TT>(allocator);

  if (arg_lists.empty()) {
    return nullptr;
  }

  size_t batch_size = arg_lists.size();

  std::vector<TT*> ts_ptrs;
  ts_ptrs.reserve(arg_lists.size());
  std::vector<int64_t> shape = arg_lists[0][axis]->Shape();
  for (auto& ts : arg_lists) {
    if (shape != ts[axis]->Shape()) {
      return nullptr;
    }
    ts_ptrs.push_back(static_cast<TT*>(ts[axis]));
  }

  std::vector<int64_t> output_shape = shape;
  output_shape.insert(output_shape.begin(), batch_size);

  char* buff = reinterpret_cast<char*>(output->Allocate(output_shape));
  for (size_t i = 0; i < batch_size; ++i) {
    auto ts = ts_ptrs[i];
    const char* ts_buff = reinterpret_cast<const char*>(ts->DataRaw());
    auto ts_size = ts->SizeInBytes();
    std::memcpy(buff + i * ts_size, ts_buff, ts_size);
  }

  return output.release();
}

std::tuple<OrtxStatus, ProcessorResult> ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data,
                                                                   ortc::Tensor<float>** pixel_values,
                                                                   ortc::Tensor<int64_t>** image_sizes,
                                                                   ortc::Tensor<int64_t>** num_img_takens) const {
  ProcessorResult r;
  std::vector<TensorArgs> inputs;
  inputs.resize(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    ts_input.push_back(std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data()).release());
  }

  std::vector<TensorArgs> outputs;

  std::vector<Operation*> ops(operations_.size());
  std::transform(operations_.begin(), operations_.end(), ops.begin(), [](auto& op) { return op.get(); });
  OrtxRunner runner(allocator_, ops.data(), ops.size());
  auto status = runner.Run(inputs, outputs);
  if (!status.IsOk()) {
    return {status, r};
  }

  // clear the input tensors
  for (auto& input : inputs) {
    for (auto& ts : input) {
      std::unique_ptr<ortc::TensorBase>(ts).reset();
    }
  }

  operations_.back()->ResetTensors(allocator_);

  *pixel_values = r.pixel_values = StackTensor<float>(outputs, 0, allocator_);
  *image_sizes = r.image_sizes = StackTensor<int64_t>(outputs, 1, allocator_);
  *num_img_takens = r.num_img_tokens = StackTensor<int64_t>(outputs, 2, allocator_);

  return {status, std::move(r)};
}

OrtxStatus ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data, TensorResult& r) const {
  std::vector<TensorArgs> inputs;
  inputs.resize(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    ts_input.push_back(std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data()).release());
  }

  std::vector<TensorArgs> outputs;
  std::vector<Operation*> ops(operations_.size());
  std::transform(operations_.begin(), operations_.end(), ops.begin(), [](auto& op) { return op.get(); });
  OrtxRunner runner(allocator_, ops.data(), ops.size());
  auto status = runner.Run(inputs, outputs);
  if (!status.IsOk()) {
    return status;
  }

  // clear the input tensors
  for (auto& input : inputs) {
    for (auto& ts : input) {
      std::unique_ptr<ortc::TensorBase>(ts).reset();
    }
  }

  auto img_result = operations_.back()->AllocateOutputs(allocator_);
  status = OrtxRunner::StackTensors(outputs, img_result, allocator_);
  operations_.back()->ResetTensors(allocator_);
  if (status.IsOk()) {
    r.SetTensors(std::move(img_result));
  }

  return status;
}

void ImageProcessor::ClearOutputs(ProcessorResult* r) {
  if (r->pixel_values) {
    std::unique_ptr<ortc::TensorBase>(r->pixel_values).reset();
    r->pixel_values = nullptr;
  }

  if (r->image_sizes) {
    std::unique_ptr<ortc::TensorBase>(r->image_sizes).reset();
    r->image_sizes = nullptr;
  }

  if (r->num_img_tokens) {
    std::unique_ptr<ortc::TensorBase>(r->num_img_tokens).reset();
    r->num_img_tokens = nullptr;
  }
}

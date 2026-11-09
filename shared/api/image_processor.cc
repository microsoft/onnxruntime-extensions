// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <string_view>

#include "nlohmann/json.hpp"
#include "file_sys.h"

#include "vision/decode_image.hpp"
#include "image_processor.h"
#include "c_api_utils.hpp"

#include "image_transforms.hpp"
#include "image_transforms_phi_3.hpp"
#include "image_transforms_mllama.hpp"
#include "image_transforms_phi_4.hpp"
#include "image_transforms_qwen2_5.hpp"
#include "image_transforms_gemma4.hpp"
#include "image_transforms_pixtral.hpp"

namespace ort_extensions {
std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(
    const std::initializer_list<const char*>& image_paths) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths.begin(), image_paths.end());
}

std::tuple<std::unique_ptr<ImageRawData[]>, size_t> LoadRawImages(const char* image_paths[], size_t num_images) {
  return ort_extensions::LoadRawData<const char* const*, ImageRawData>(image_paths, image_paths + num_images);
}
}  // namespace ort_extensions

using namespace ort_extensions;
using json = nlohmann::json;

Operation::KernelRegistry ImageProcessor::kernel_registry_ = {
    {"DecodeImage", []() { return CreateKernelInstance(&ort_extensions::DecodeImage::Compute); }},
    {"Resize", []() { return CreateKernelInstance(&Resize::Compute); }},
    {"Rescale", []() { return CreateKernelInstance(&Rescale::Compute); }},
    {"Normalize", []() { return CreateKernelInstance(&Normalize::Compute); }},
    {"CenterCrop", []() { return CreateKernelInstance(&CenterCrop::Compute); }},
    {"ConvertRGB", []() { return CreateKernelInstance(convert_to_rgb); }},
    {"Permute3D", []() { return CreateKernelInstance(&Permute3D::Compute); }},
    {"Phi3ImageTransform", []() { return CreateKernelInstance(phi3_hd_transform); }},
    {"Llama3ImageTransform", []() { return CreateKernelInstance(&Llama3ImageTransform::Compute); }},
    {"Phi4VisionDynamicPreprocess", []() { return CreateKernelInstance(&Phi4VisionDynamicPreprocess::Compute); }},
    {"Phi4VisionProcessor", []() { return CreateKernelInstance(&Phi4VisionProcessor::Compute); }},
    {"PatchImage", []() { return CreateKernelInstance(&PatchImage::Compute); }},
    {"Gemma4ImageTransform", []() { return CreateKernelInstance(&Gemma4ImageTransform::Compute); }},
    {"PixtralImageSizes", []() { return CreateKernelInstance(&PixtralImageSizes::Compute); }}};  // NOLINT

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

  if (auto iter = processor_root.find("name"); iter != processor_root.end()) {
    name_ = iter->get<std::string>();
  }

  if (auto iter = processor_root.find("output_aligner"); iter != processor_root.end()) {
    output_aligner_ = iter->get<std::string>();
  }

  auto transforms = processor_root.at("transforms");
  if (!transforms.is_array() || transforms.empty()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: transforms field is missing."};
  }

  ORTX_RETURN_IF_ERROR(op_plan_.Init(transforms, kernel_registry_));
  image_decode_.reset();
  image_transforms_.clear();
  if (transforms.size() >= 3 && transforms[0].at("operation").at("type") == "DecodeImage" &&
      transforms[1].at("operation").at("type") == "ConvertRGB" &&
      transforms[2].at("operation").at("type") == "Resize") {
    image_decode_ = CreateKernelInstance(&ort_extensions::DecodeImage::Compute);
    ORTX_RETURN_IF_ERROR(image_decode_->Init(transforms[0].at("operation").value("attrs", json::object()).dump()));
    auto image_transforms = transforms;
    image_transforms.erase(image_transforms.begin());
    image_transforms_ = image_transforms.dump();
  }
  frame_plan_.reset();
  frame_patch_.reset();
  frame_transforms_.clear();
  const auto first_transform = transforms.front().at("operation").at("type").get<std::string>();
  if ((first_transform == "Resize" || first_transform == "Rescale") &&
      transforms.back().at("operation").at("type") == "PatchImage" && transforms.size() > 1) {
    frame_patch_ = CreateKernelInstance(&PatchImage::Compute);
    ORTX_RETURN_IF_ERROR(frame_patch_->Init(transforms.back().at("operation").value("attrs", json::object()).dump()));
    transforms.erase(transforms.end() - 1);
    frame_transforms_ = transforms.dump();
    frame_plan_ = std::make_unique<ExecutionPlan>();
    ORTX_RETURN_IF_ERROR(frame_plan_->Init(transforms, kernel_registry_));
  }
  return {};
}

ImageProcessor::ImageProcessor() : OrtxObjectImpl(kOrtxKindProcessor) {}

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

  std::vector<TensorArgs> inputs(image_data.size());
  std::vector<TensorPtr> input_tensor_objects(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    input_tensor_objects[i] = std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data());
    ts_input.push_back(input_tensor_objects[i].get());
  }

  std::vector<TensorArgs> outputs;
  OrtxRunner runner(op_plan_);
  auto status = runner.Run(inputs, outputs);
  if (!status.IsOk()) {
    return {status, r};
  }

  input_tensor_objects.clear();
  *pixel_values = r.pixel_values = StackTensor<float>(outputs, 0, runner.GetAllocator());
  *image_sizes = r.image_sizes = StackTensor<int64_t>(outputs, 1, runner.GetAllocator());
  *num_img_takens = r.num_img_tokens = StackTensor<int64_t>(outputs, 2, runner.GetAllocator());

  return {status, std::move(r)};
}

OrtxStatus ImageProcessor::PreProcessFrames(ort_extensions::span<const OrtxFrame> frames,
                                            const OrtxFramePreProcessOptions* options, TensorResult& r) const {
  if (!frame_plan_ || frames.size() == 0) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: decoded RGB frame pipeline is required"};
  }
  ExecutionPlan resolved_plan;
  const ExecutionPlan* plan = frame_plan_.get();
  if (options != nullptr) {
    auto transforms = json::parse(frame_transforms_);
    auto& resize = transforms.front().at("operation");
    if (resize.at("type") != "Resize")
      return {kOrtxErrorInvalidArgument, "[ImageProcessor]: resolved frame size requires an initial Resize"};
    auto& attrs = resize["attrs"];
    if (attrs.is_null()) attrs = json::object();
    attrs["height"] = options->height;
    attrs["width"] = options->width;
    attrs["smart_resize"] = 0;
    attrs["keep_aspect_ratio"] = 0;
    ORTX_RETURN_IF_ERROR(resolved_plan.Init(transforms, kernel_registry_));
    plan = &resolved_plan;
  }
  std::vector<TensorArgs> inputs(frames.size());
  std::vector<TensorPtr> input_tensors(frames.size());
  for (size_t i = 0; i < frames.size(); ++i) {
    const auto& frame = frames[i];
    input_tensors[i] = std::make_unique<ortc::Tensor<uint8_t>>(std::vector<int64_t>{frame.height, frame.width, 3},
                                                               const_cast<void*>(frame.data));
    inputs[i].push_back(input_tensors[i].get());
  }
  std::vector<TensorArgs> outputs;
  struct OutputGuard {
    std::vector<TensorArgs>& outputs;
    ~OutputGuard() {
      for (auto& tensors : outputs)
        for (auto* tensor : tensors) delete tensor;
    }
  } guard{outputs};
  OrtxRunner runner(*plan);
  ORTX_RETURN_IF_ERROR(runner.Run(inputs, outputs));
  for (const auto& tensors : outputs) {
    if (tensors.size() != 1 || tensors[0]->Type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        tensors[0]->Shape().size() != 3 || tensors[0]->Shape()[2] != 3) {
      return {kOrtxErrorInvalidArgument, "[ImageProcessor]: frame transforms must produce HWC float RGB"};
    }
  }
  std::unique_ptr<ortc::Tensor<float>> stacked(StackTensor<float>(outputs, 0, runner.GetAllocator()));
  if (!stacked) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: processed frame geometry must agree"};
  }
  auto pixels = std::make_unique<ortc::Tensor<float>>(runner.GetAllocator());
  auto grid = std::make_unique<ortc::Tensor<int64_t>>(runner.GetAllocator());
  TensorArgs patch_input{stacked.get()}, patch_output{pixels.get(), grid.get()};
  ORTX_RETURN_IF_ERROR(frame_patch_->Invoke(patch_input, patch_output));
  std::vector<TensorPtr> result;
  result.reserve(2);
  result.push_back(std::move(pixels));
  result.push_back(std::move(grid));
  r.SetTensors(std::move(result));
  return {};
}

OrtxStatus ImageProcessor::PreProcess(ort_extensions::span<ImageRawData> image_data, TensorResult& r,
                                      ort_extensions::span<const OrtxImagePreProcessOptions> options) const {
  if (options.size() != 0 && (!image_decode_ || options.size() != image_data.size()))
    return {kOrtxErrorInvalidArgument,
            "[ImageProcessor]: resolved images require a DecodeImage/ConvertRGB/Resize prefix"};
  std::vector<TensorArgs> inputs(image_data.size());
  std::vector<TensorPtr> input_tensor_objects(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    input_tensor_objects[i] = std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data());
    ts_input.push_back(input_tensor_objects[i].get());
  }

  std::vector<TensorArgs> outputs;
  struct OutputGuard {
    std::vector<TensorArgs>& outputs;
    ~OutputGuard() {
      for (auto& tensors : outputs)
        for (auto* tensor : tensors) delete tensor;
    }
  } guard{outputs};
  OrtxRunner runner(op_plan_);
  OrtxStatus status;
  if (options.size() == 0) {
    status = runner.Run(inputs, outputs);
  } else {
    auto transforms = json::parse(image_transforms_);
    auto& resize = transforms[1].at("operation")["attrs"];
    if (resize.is_null()) resize = json::object();
    resize["smart_resize"] = 0;
    resize["keep_aspect_ratio"] = 0;
    for (size_t image = 0; image < image_data.size(); ++image) {
      auto decoded = image_decode_->AllocateOutput(runner.GetAllocator());
      TensorPtr decoded_owner{decoded[0]};
      ORTX_RETURN_IF_ERROR(image_decode_->Invoke(inputs[image], decoded));
      if (decoded[0]->Shape() != std::vector<int64_t>{options[image].source_height, options[image].source_width, 3})
        return {kOrtxErrorInvalidArgument, "[ImageProcessor]: decoded dimensions differ from the resolved source"};
      resize["height"] = options[image].height;
      resize["width"] = options[image].width;
      ExecutionPlan plan;
      ORTX_RETURN_IF_ERROR(plan.Init(transforms, kernel_registry_));
      OrtxRunner resolved_runner(plan);
      std::vector<TensorArgs> single_input{decoded};
      ORTX_RETURN_IF_ERROR(resolved_runner.Run(single_input, outputs));
    }
  }
  if (!status.IsOk()) {
    return status;
  }

  // clear the input tensors
  input_tensor_objects.clear();

  std::vector<TensorPtr> img_result = op_plan_.AllocateOutputs(runner.GetAllocator());
  ORTX_RETURN_IF_ERROR(OrtxRunner::StackTensors(outputs, img_result, runner.GetAllocator()));

  if (output_aligner_ == "phi4-vision-aligner") {
    status = Phi4VisionProcessor::AlignOutputs(img_result);
  }

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

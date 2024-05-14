// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nlohmann/json.hpp"
#include "image_processor.h"
#include "cv2/imgcodecs/imdecode.hpp"
#include "image_transforms.hpp"

using namespace ort_extensions;
using json = nlohmann::json;

std::unordered_map<std::string_view, std::function<std::unique_ptr<KernelClass>()>> Operation::kernel_registry_ = {
    {"DecodeImage", []() { return DefineKernelFunction(image_decoder); }},
    {"ConvertRGB", []() { return DefineKernelFunction(convert_to_rgb); }},
    {"Phi3ImageTransform", []() { return DefineKernelFunction(phi3_hd_transform); }},
};

OrtxStatus Operation::Init(std::string_view op_def) {
  // parse the op_def by json
  auto full_json = json::parse(op_def);
  if (!full_json.is_object()) {
    return {kOrtxErrorInvalidArgument, "[Operation]: failed to parse op_def."};
  }

  auto op_json = full_json.at("operation");

  auto op_name = op_json.at("name").get<std::string>();
  if (op_name.empty()) {
    return {kOrtxErrorInvalidArgument, "[Operation]: name field is missing."};
  }

  auto op_type = op_json.at("type").get<std::string>();
  if (op_type.empty()) {
    return {kOrtxErrorInvalidArgument, "[Operation]: type field is missing."};
  }

  auto kernel_iter = kernel_registry_.find(op_type);
  if (kernel_iter == kernel_registry_.end()) {
    return {kOrtxErrorInvalidArgument, "[Operation]: type is not supported."};
  }

  op_name_ = op_name;
  kernel_ = kernel_iter->second();

  /*
    if (op_json.contains("attrs")) {
      auto attrs = op_json.at("attrs");
      attrs.get_to(attributes_);
    }
  */
  return {};
}

void Operation::ResetTensors(ortc::IAllocator* allocator) {
  outputs_.clear();
}

Operation::~Operation() {
  ResetTensors(allocator_);
}

class OrtxRunner {
 public:
  OrtxRunner(ortc::IAllocator* allocator, Operation** ops, size_t op_num)
      : allocator_(allocator), ops_(ops, ops + op_num) {}

  template <typename IT, typename OT>  // batch input/output containter
  OrtxStatus Run(IT& inputs, OT& outputs) {
    OT output_list;
    Operation* last_op = nullptr;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (last_op != nullptr) {
        last_op->ResetTensors(allocator_);
      }

      auto& input = *(inputs.begin() + i);
      for (auto& op : ops_) {
        auto [status, ts_output] = op->Apply(allocator_, input);
        if (status.IsOk()) {
          if (op == ops_.back()) {
            output_list.push_back(ts_output);
          } else {
            input = ts_output;
          }
        } else {
          return status;
        }

        last_op = op;
      }
    }

    outputs = std::move(output_list);
    return {};
  }

 private:
  std::vector<Operation*> ops_;
  ortc::IAllocator* allocator_;
};

OrtxStatus ImageProcessor::Init(std::string_view processor_def) {
  // pase the processor_def by json
  auto proc_json = json::parse(processor_def, nullptr, false);
  if (proc_json.is_discarded()) {
    return {kOrtxErrorInvalidArgument, "[ImageProcessor]: failed to parse processor_def."};
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
    auto op = std::make_unique<Operation>();
    auto status = op->Init(mod_iter->dump());
    if (!status.IsOk()) {
      return status;
    }

    operations_.push_back(std::move(op));
  }

  return {};
}

class SimpleAllocator : public ortc::IAllocator {
 public:
  void* Alloc(size_t size) override {
    return std::make_unique<char[]>(size).release();
  }

  void Free(void* p) override {
    std::unique_ptr<char[]> ptr(static_cast<char*>(p));
    ptr.reset();
  }
};

static SimpleAllocator g_allocator_;

ImageProcessor::ImageProcessor()
    : allocator_(&g_allocator_), OrtxObjectImpl(kOrtxKindProcessor) {
}

std::tuple<OrtxStatus, ProcessorResult>
ImageProcessor::PreProcess(
    ort_extensions::span<ImageRawData> image_data,
    ortc::Tensor<float>** pixel_values,
    ortc::Tensor<int64_t>** image_sizes,
    ortc::Tensor<int64_t>** num_img_takens) {

  ProcessorResult r;
  r.last_operation_ = operations_.back().get();

  std::vector<TensorArgs> inputs;
  inputs.resize(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_input = inputs[i];
    ImageRawData& image = image_data[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(image.size())};
    ts_input.push_back(std::make_unique<ortc::Tensor<uint8_t>>(shape, image.data()).release());
  }

  std::vector<TensorArgs> outputs;
  outputs.resize(image_data.size());
  for (size_t i = 0; i < image_data.size(); ++i) {
    auto& ts_output = outputs[i];
    ts_output.push_back(std::make_unique<ortc::Tensor<uint8_t>>(allocator_).release());
    ts_output.push_back(std::make_unique<ortc::Tensor<int64_t>>(allocator_).release());
    ts_output.push_back(std::make_unique<ortc::Tensor<int64_t>>(allocator_).release());
  }

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
      delete ts;
    }
  }

  *pixel_values = static_cast<ortc::Tensor<float>*>(outputs[0][0]);
  *image_sizes = static_cast<ortc::Tensor<int64_t>*>(outputs[0][1]);
  *num_img_takens = static_cast<ortc::Tensor<int64_t>*>(outputs[0][2]);

  return {status, r};
}

void ImageProcessor::ClearOutputs(ProcessorResult* r) {
  if (r != nullptr) {
    if (r->last_operation_ != nullptr) {
      r->last_operation_->ResetTensors(allocator_);
    }
  }
}

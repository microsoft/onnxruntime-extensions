// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speech_extractor.h"

#include "audio/audio_decoder.h"
#include "speech_features.hpp"

using namespace ort_extensions;

Operation::KernelRegistry SpeechFeatureExtractor::kernel_registry_ = {
  {"AudioDecoder", []() { return CreateKernelInstance(&AudioDecoder::ComputeNoOpt); }},
  {"AudioDecoderEx", []() { return CreateKernelInstance(&AudioDecoder::ComputeNoOpt2); }},
  {"STFTNorm", []() { return CreateKernelInstance(&SpeechFeatures::STFTNorm); }},
  {"LogMelSpectrum", []() { return CreateKernelInstance(&LogMel::Compute); }},
  {"Phi4AudioEmbed", []() { return CreateKernelInstance(&Phi4AudioEmbed::Compute); }}
};

SpeechFeatureExtractor::SpeechFeatureExtractor()
  : OrtxObjectImpl(extObjectKind_t::kOrtxKindFeatureExtractor) {}

OrtxStatus SpeechFeatureExtractor::Init(std::string_view extractor_def) {
  std::string fe_def_str;
  if (extractor_def.size() >= 5 && extractor_def.substr(extractor_def.size() - 5) == ".json") {
    std::ifstream ifs = path({extractor_def.data(), extractor_def.size()}).open();
    if (!ifs.is_open()) {
      return {kOrtxErrorInvalidArgument, std::string("[ImageProcessor]: failed to open ") + std::string(extractor_def)};
    }
    fe_def_str = std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    extractor_def = fe_def_str.c_str();
  }

  // pase the extraction_def by json
  auto fe_json = json::parse(extractor_def, nullptr, false);
  if (fe_json.is_discarded()) {
    return {kOrtxErrorInvalidArgument, "[SpeechFeatureExtractor]: failed to parse extractor json configuration."};
  }

  auto fe_root = fe_json.at("feature_extraction");
  if (!fe_root.is_object()) {
    return {kOrtxErrorInvalidArgument, "[SpeechFeatureExtractor]: feature_extraction field is missing."};
  }

  auto op_sequence = fe_root.at("sequence");
  if (!op_sequence.is_array() || op_sequence.empty()) {
    return {kOrtxErrorInvalidArgument, "[SpeechFeatureExtractor]: sequence field is missing."};
  }

  return op_plan_.Init(op_sequence, kernel_registry_);
}

OrtxStatus SpeechFeatureExtractor::DoCall(ort_extensions::span<AudioRawData> raw_speech,
                                          std::unique_ptr<ortc::Tensor<float>>& log_mel) const {
  // setup the input tensors
  std::vector<TensorArgs> inputs;
  inputs.resize(raw_speech.size());
  for (size_t i = 0; i < raw_speech.size(); ++i) {
    auto& ts_input = inputs[i];
    AudioRawData& speech = raw_speech[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(speech.size())};
    ts_input.push_back(std::make_unique<ortc::Tensor<uint8_t>>(shape, speech.data()).release());
  }

  std::vector<TensorArgs> outputs;
  // std::vector<Operation*> ops(operations_.size());
  // std::transform(operations_.begin(), operations_.end(), ops.begin(), [](auto& op) { return op.get(); });
  // OrtxRunner runner(allocator_, ops.data(), ops.size());
  OrtxRunner runner(op_plan_);
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

  auto results = op_plan_.AllocateOutputs(runner.GetAllocator());
  status = OrtxRunner::StackTensors(outputs, results, runner.GetAllocator());
  if (status.IsOk()) {
    log_mel.reset(static_cast<ortc::Tensor<float>*>(results[0].release()));
  }

  return status;
}

OrtxStatus
SpeechFeatureExtractor::Preprocess(ort_extensions::span<AudioRawData> raw_speech, TensorResult& r) const {
                                              // std::unique_ptr<ortc::Tensor<float>>& audio_embeds,
                                              // std::unique_ptr<ortc::Tensor<int64_t>>& audio_embed_sizes) const {
  // setup the input tensors
  std::vector<TensorArgs> inputs;
  inputs.resize(raw_speech.size());
  for (size_t i = 0; i < raw_speech.size(); ++i) {
    auto& ts_input = inputs[i];
    AudioRawData& speech = raw_speech[i];
    std::vector<int64_t> shape = {static_cast<int64_t>(speech.size())};
    ts_input.push_back(std::make_unique<ortc::Tensor<uint8_t>>(shape, speech.data()).release());
  }

  std::vector<TensorArgs> outputs;
  // std::vector<Operation*> ops(operations_.size());
  // std::transform(operations_.begin(), operations_.end(), ops.begin(), [](auto& op) { return op.get(); });
  // OrtxRunner runner(allocator_, ops.data(), ops.size());
  OrtxRunner runner(op_plan_);
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

  auto results = op_plan_.AllocateOutputs(runner.GetAllocator());
  status = OrtxRunner::StackTensors(outputs, results, runner.GetAllocator());
  if (status.IsOk()) {
    r.SetTensors(std::move(results));
  }

  return status;
}

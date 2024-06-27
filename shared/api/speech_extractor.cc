// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "speech_extractor.h"

#include "c_api_utils.hpp"
#include "speech_features.hpp"

using namespace ort_extensions;

OrtxStatus log_mel_spectrum(const ortc::Tensor<float>& stfm_norm, ortc::Tensor<float>& logmel) {
  // magnitudes = stft_norm[:, :, :-1]
  // mel_spec = self.mel_filters @ magnitudes
  // log_spec = torch.clamp(mel_spec, min=1e-10).log10()
  // spec_min = log_spec.max() - 8.0
  // log_spec = torch.maximum(log_spec, spec_min)
  // spec_shape = log_spec.shape
  // padding_spec = torch.ones(spec_shape[0],
  //                           spec_shape[1],
  //                           self.n_samples // self.hop_length - spec_shape[2],
  //                           dtype=torch.float)
  // padding_spec *= spec_min
  // log_spec = torch.cat((log_spec, padding_spec), dim=2)
  // log_spec = (log_spec + 4.0) / 4.0
  // return log_spec

  return {};
}

Operation::KernelRegistry SpeechFeatureExtractor::kernel_registry_ = {
    {"AudioDecoder", []() { return CreateKernelInstance(&AudioDecoder::ComputeNoOpt); }},
    {"STFTNorm", []() { return CreateKernelInstance(&AudioFeatures::STFTNorm); }},
    {"LogMelSpectrum", []() { return CreateKernelInstance(log_mel_spectrum); }},
};

SpeechFeatureExtractor::SpeechFeatureExtractor()
    : OrtxObjectImpl(extObjectKind_t::kOrtxKindFeatureExtractor), allocator_(&CppAllocator::Instance()) {}

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

  operations_.reserve(op_sequence.size());
  for (auto mod_iter = op_sequence.begin(); mod_iter != op_sequence.end(); ++mod_iter) {
    auto op = std::make_unique<Operation>(kernel_registry_);
    auto status = op->Init(mod_iter->dump());
    if (!status.IsOk()) {
      return status;
    }

    operations_.push_back(std::move(op));
  }

  return {};
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

  auto results = operations_.back()->AllocateOutputs(allocator_);
  status = OrtxRunner::StackTensors(outputs, results, allocator_);
  if (status.IsOk()) {
    log_mel.reset(static_cast<ortc::Tensor<float>*>(results[0].release()));
    operations_.back()->ResetTensors(allocator_);
  }

  return status;
}

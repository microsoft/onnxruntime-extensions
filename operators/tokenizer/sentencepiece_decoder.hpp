// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"

struct KernelSentencepieceDecoder : BaseKernel {
  KernelSentencepieceDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
    std::string model_blob = ort_.KernelInfoGetAttribute<std::string>(&info, "model");
    sentencepiece::ModelProto model_proto;
    model_proto.ParseFromArray(model_blob.data(), static_cast<int>(model_blob.size()));
    sentencepiece::util::Status status = tokenizer_.Load(model_proto);
    if (!status.ok()) {
      ORTX_CXX_API_THROW(MakeString("Failed to create SentencePieceProcessor instance. Error code is ",
                                    (int)status.code(), ". Message is '", status.error_message(), "'."),
                         ORT_INVALID_PROTOBUF);
    }
  }

  void Compute(const ortc::TensorT<int64_t>& ids,
               ortc::TensorT<std::string>& output) {
    const int64_t* p_ids = ids.Data();
    auto& ids_dim = ids.Shape();

    if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
      ORTX_CXX_API_THROW("[SentencePieceDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
    }

    std::string decoded_string;
    std::vector<int64_t> output_dim = {1};
    std::vector<int> tids;
    std::transform(p_ids, p_ids + ids.NumerOfElement(),
                   std::back_inserter(tids),
                   [](auto _id) { return static_cast<int>(_id); });
    auto status = tokenizer_.Decode(tids, &decoded_string);
    if (!status.ok()) {
      ORTX_CXX_API_THROW("[SentencePieceDecoder] model decoding failed.", ORT_RUNTIME_EXCEPTION);
    }

    std::vector<std::string> result = {decoded_string};
    output.SetStringOutput(0, result, output_dim);
  }

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

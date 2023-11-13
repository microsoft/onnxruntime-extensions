// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"

struct KernelSentencepieceDecoder {
  OrtStatusPtr OnModelAttach(const OrtApi& api, const OrtKernelInfo& info) {
    std::string model_blob;
    ORTX_RETURN_IF_ERROR(OrtW::GetOpAttribute(info, "model", model_blob));

    sentencepiece::ModelProto model_proto;
    model_proto.ParseFromArray(model_blob.data(), static_cast<int>(model_blob.size()));
    sentencepiece::util::Status status = tokenizer_.Load(model_proto);
    if (!status.ok()) {
      return OrtW::CreateStatus(MakeString("Failed to create SentencePieceProcessor instance. Error code is ",
                                           (int)status.code(), ". Message is '", status.error_message(), "'."),
                                ORT_INVALID_PROTOBUF);
    }

    return nullptr;
  }

  OrtStatusPtr Compute(const ortc::Tensor<int64_t>& ids,
                       ortc::Tensor<std::string>& output) const {
    const int64_t* p_ids = ids.Data();
    auto& ids_dim = ids.Shape();

    if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
      return OrtW::CreateStatus("[SentencePieceDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
    }

    std::string decoded_string;
    std::vector<int64_t> output_dim = {1};
    std::vector<int> tids;
    std::transform(p_ids, p_ids + ids.NumberOfElement(),
                   std::back_inserter(tids),
                   [](auto _id) { return static_cast<int>(_id); });
    auto status = tokenizer_.Decode(tids, &decoded_string);
    if (!status.ok()) {
      return OrtW::CreateStatus("[SentencePieceDecoder] model decoding failed.", ORT_RUNTIME_EXCEPTION);
    }

    std::vector<std::string> result = {decoded_string};
    output.SetStringOutput(result, output_dim);
    return nullptr;
  }

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

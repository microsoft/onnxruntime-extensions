// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ocos.h"
#include "string_utils.h"
#include "string_tensor.h"
#include "sentencepiece_processor.h"
#include "sentencepiece_model.pb.h"

struct KernelSentencepieceDecoder : BaseKernel {
  KernelSentencepieceDecoder(const OrtApi& api, const OrtKernelInfo* info) : BaseKernel(api, info) {
    std::string model_blob = ort_.KernelInfoGetAttribute<std::string>(info, "model");
    sentencepiece::ModelProto model_proto;
    model_proto.ParseFromArray(model_blob.data(), static_cast<int>(model_blob.size()));
    sentencepiece::util::Status status = tokenizer_.Load(model_proto);
    if (!status.ok()){
      ORT_CXX_API_THROW(MakeString(
                            "Failed to create SentencePieceProcessor instance. Error code is ",
                            (int)status.code(), ". Message is '", status.error_message(), "'."),
                        ORT_INVALID_PROTOBUF);
    }
  }

  void Compute(OrtKernelContext* context) {
    const OrtValue* ids = ort_.KernelContext_GetInput(context, 0);
    const int64_t* p_ids = ort_.GetTensorData<int64_t>(ids);
    OrtTensorDimensions ids_dim(ort_, ids);

    if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
      ORT_CXX_API_THROW("[SentencePieceDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
    }

    auto count = ids_dim[0];
    std::string decoded_string;
    std::vector<int64_t> output_dim = {1};
    std::vector<int> tids;
    std::transform(p_ids, p_ids + count,
                   std::back_inserter(tids),
                   [](auto _id) { return static_cast<int>(_id); });
    auto status = tokenizer_.Decode(tids, &decoded_string);
    if (!status.ok()){
      ORT_CXX_API_THROW("[SentencePieceDecoder] model decoding failed.", ORT_RUNTIME_EXCEPTION);
    }

    std::vector<std::string> result = {decoded_string};
    OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());
    FillTensorDataString(api_, ort_, context, result, output);
  }

 private:
  sentencepiece::SentencePieceProcessor tokenizer_;
};

struct CustomOpSentencepieceDecoder : Ort::CustomOpBase<CustomOpSentencepieceDecoder, KernelSentencepieceDecoder> {
  void* CreateKernel(const OrtApi& api, const OrtKernelInfo* info) const {
    return CreateKernelImpl(api, info);
  }

  const char* GetName() const {
    return "SentencepieceDecoder";
  }

  size_t GetInputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetInputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
  }

  size_t GetOutputTypeCount() const {
    return 1;
  }

  ONNXTensorElementDataType GetOutputType(size_t index) const {
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
  }
};

#include "bert_tokenizer_decoder.hpp"

BertTokenizerDecoder::BertTokenizerDecoder(std::string vocab, ustring unk_token, ustring sep_token, ustring pad_token,
                                           ustring cls_token, ustring mask_token, ustring suffix_indicator) : unk_token_(unk_token), suffix_indicator_(suffix_indicator) {
  auto tokens = SplitString(vocab, "\n", true);
  vocab_.reserve(tokens.size());
  for (int i = 0; i < tokens.size(); i++) {
    ustring token(tokens[i]);
    if (token == unk_token) {
      unk_token_id_ = i;
    }
    if (token == sep_token) {
      sep_token_id_ = i;
    }
    if (token == pad_token) {
      sep_token_id_ = i;
    }
    if (token == cls_token) {
      cls_token_id_ = i;
    }
    if (token == mask_token) {
      mask_token_id_ = i;
    }

    if (token.rfind(suffix_indicator_, 0) == 0) {
      vocab_.emplace_back(token.substr(suffix_indicator.size(), token.size() - suffix_indicator.size()));
      is_substr_.push_back(true);
    } else {
      vocab_.push_back(token);
      is_substr_.push_back(false);
    }
  }
}

ustring BertTokenizerDecoder::Decode(const std::vector<int64_t>& ids) {
  ustring result;
  for (auto id : ids) {
    if (id == unk_token_id_ || id == sep_token_id_ || id == pad_token_id_ || id == cls_token_id_ || id == mask_token_id_) {
      continue;
    }

    // deal with unk ids
    if (id >= vocab_.size() || id < 0) {
      if (result.empty()) {
        result.push_back(U' ');
      }
      result.append(unk_token_);
      continue;
    }

    // skip first substr
    if (result.empty() && is_substr_[id]) {
      continue;
    }

    if (!result.empty() && !is_substr_[id]) {
      result.push_back(U' ');
    }

    result.append(vocab_[id]);
  }

  return result;
}

KernelBertTokenizerDecoder::KernelBertTokenizerDecoder(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(info, "vocab_file");
  std::string unk_token = TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
  std::string sep_token = TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
  std::string pad_token = TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
  std::string cls_token = TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
  std::string mask_token = TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
  std::string suffix_indicator = TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));
  use_indices_ = TryToGetAttributeWithDefault("use_indices", false);

  decoder_ = std::make_shared<BertTokenizerDecoder>(vocab, ustring(unk_token), ustring(sep_token), ustring(pad_token),
                                                    ustring(cls_token), ustring(mask_token), ustring(suffix_indicator));
}

void KernelBertTokenizerDecoder::Compute(OrtKernelContext* context) {
  const OrtValue* ids = ort_.KernelContext_GetInput(context, 0);
  const int64_t* p_ids = ort_.GetTensorData<int64_t>(ids);
  OrtTensorDimensions ids_dim(ort_, ids);

  if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
    ORT_CXX_API_THROW("[BertTokenizerDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
  }

  //  const int64_t* p_row_indices = ort_row_indices_dim.empty() ? nullptr : ort_.GetTensorData<int64_t>(ort_row_indices);
  const OrtValue* positions = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions positions_dim(ort_, positions);
  if (use_indices_ &&
      (!(positions_dim.empty() ||
         (positions_dim.Size() == 0) ||
         (positions_dim.size() == 2 && positions_dim[1] == 2)))) {
    ORT_CXX_API_THROW("[BertTokenizerDecoder]: Expect positions empty or a [n, 2] matrix when use indices", ORT_INVALID_GRAPH);
  }

  const int64_t* p_positions =  positions_dim.Size() == 0 ? nullptr : ort_.GetTensorData<int64_t>(positions);

  std::vector<ustring> result;
  std::vector<int64_t> output_dim(1);
  if (!use_indices_) {
    result.push_back(decoder_->Decode(std::vector<int64_t>(p_ids, p_ids + ids_dim.Size())));
    output_dim[0] = 1;
  } else {
    if (p_positions != nullptr) {
      for (int i = 0; i < positions_dim[0]; i++) {
        int64_t start = p_positions[2 * i];
        int64_t end = p_positions[2 * i + 1];

        result.push_back(decoder_->Decode(std::vector<int64_t>(p_ids + start, p_ids + end)));
      }
      output_dim[0] = positions_dim[0];
    }
  }
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());

  FillTensorDataString(api_, ort_, context, result, output);
}

void* CustomOpBertTokenizerDecoder::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBertTokenizerDecoder(api, info);
};

const char* CustomOpBertTokenizerDecoder::GetName() const { return "BertTokenizerDecoder"; };

size_t CustomOpBertTokenizerDecoder::GetInputTypeCount() const {
  return 2;
};

ONNXTensorElementDataType CustomOpBertTokenizerDecoder::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};

size_t CustomOpBertTokenizerDecoder::GetOutputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBertTokenizerDecoder::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

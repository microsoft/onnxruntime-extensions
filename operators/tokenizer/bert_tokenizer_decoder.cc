#include "bert_tokenizer_decoder.hpp"

BertTokenizerDecoder::BertTokenizerDecoder(
    std::string vocab,
    std::string unk_token,
    std::string sep_token,
    std::string pad_token,
    std::string cls_token,
    std::string mask_token,
    std::string suffix_indicator) : unk_token_(unk_token),
                                    suffix_indicator_(suffix_indicator),
                                    raw_vocab_(vocab) {
  auto tokens = SplitString(raw_vocab_, "\n", true);
  vocab_.reserve(tokens.size());
  for (size_t i = 0; i < tokens.size(); i++) {
    auto& token = tokens[i];
    if (token == unk_token) {
      unk_token_id_ = static_cast<int32_t>(i);
    }
    if (token == sep_token) {
      sep_token_id_ = static_cast<int32_t>(i);
    }
    if (token == pad_token) {
      sep_token_id_ = static_cast<int32_t>(i);
    }
    if (token == cls_token) {
      cls_token_id_ = static_cast<int32_t>(i);
    }
    if (token == mask_token) {
      mask_token_id_ = static_cast<int32_t>(i);
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

std::string BertTokenizerDecoder::Decode(const std::vector<int64_t>& ids, bool skip_special_tokens, bool clean_up_tokenization_spaces) {
  std::string result;
  int64_t pre_token = -1;

  for (auto id : ids) {
    if (skip_special_tokens && (id == sep_token_id_ || id == pad_token_id_ || id == cls_token_id_ || id == mask_token_id_)) {
      continue;
    }

    // deal with unk ids
    if (id < 0 || static_cast<size_t>(id) >= vocab_.size()) {
      if (!result.empty()) {
        result.push_back(' ');
      }
      result.append(unk_token_);
      continue;
    }

    // skip first substr
    if (result.empty() && is_substr_[static_cast<size_t>(id)]) {
      continue;
    }

    // At following situations, we needn't add space
    // we needn't add a space at the beginning of the output
    // we needn't add a space when the token is a substr (such as ##ing)
    // we needn't add a space at the left or right of punctuation (such as client-side shouldn't be client - side), when clean_up_tokenization_spaces is true
    if (!(result.empty() || is_substr_[static_cast<size_t>(id)] || (clean_up_tokenization_spaces && RemoveTokenizeSpace(pre_token, id)))) {
      result.push_back(' ');
    }

    result.append(vocab_[static_cast<size_t>(id)]);
    pre_token = id;
  }

  return result;
}

bool BertTokenizerDecoder::RemoveTokenizeSpace(int64_t pre_token_id, int64_t new_token_id) {
  if (pre_token_id < 0) {
    return true;
  }

  auto pre_char = ustring(vocab_[static_cast<size_t>(pre_token_id)]).back();
  auto cur_char = ustring(vocab_[static_cast<size_t>(new_token_id)])[0];

  // normal punctuation
  if (cur_char == U'!' || cur_char == U'.' || cur_char == U'?' || cur_char == U',' || cur_char == '~' || cur_char == ':') {
    return true;
  }

  // only remove left side space
  if (cur_char == U'}' || cur_char == U']' || cur_char == U'>' || cur_char == ')') {
    return true;
  }

  // only remove right side space
  if (pre_char == U'{' || pre_char == U'[' || pre_char == U'<' || pre_char == '(' || pre_char == '$') {
    return true;
  }

  // remove both side space
  if (pre_char == U'-' || pre_char == U'\'' || pre_char == U'"' || pre_char == U'/' || pre_char == U'@' || pre_char == U'\\' ||
      cur_char == U'-' || cur_char == U'\'' || cur_char == U'"' || cur_char == U'/' || cur_char == U'@' || cur_char == U'\\') {
    return true;
  }

  // remove both space beside unicode punctuation
  if (pre_char > 128 && IsPunct(pre_char)) {
    return true;
  }

  if (cur_char > 128 && IsPunct(cur_char)) {
    return true;
  }

  return false;
}

KernelBertTokenizerDecoder::KernelBertTokenizerDecoder(const OrtApi& api, const OrtKernelInfo& info) : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(&info, "vocab_file");
  std::string unk_token = TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
  std::string sep_token = TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
  std::string pad_token = TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
  std::string cls_token = TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
  std::string mask_token = TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
  std::string suffix_indicator = TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));

  use_indices_ = TryToGetAttributeWithDefault("use_indices", false);
  skip_special_tokens_ = TryToGetAttributeWithDefault("skip_special_tokens", false);
  clean_up_tokenization_spaces_ = TryToGetAttributeWithDefault("clean_up_tokenization_spaces", true);

  decoder_ = std::make_shared<BertTokenizerDecoder>(vocab, unk_token, sep_token, pad_token,
                                                    cls_token, mask_token, suffix_indicator);
}

void KernelBertTokenizerDecoder::Compute(OrtKernelContext* context) {
  const OrtValue* ids = ort_.KernelContext_GetInput(context, 0);
  const int64_t* p_ids = ort_.GetTensorData<int64_t>(ids);
  OrtTensorDimensions ids_dim(ort_, ids);

  if (!((ids_dim.size() == 1) || (ids_dim.size() == 2 && ids_dim[0] == 1))) {
    ORTX_CXX_API_THROW("[BertTokenizerDecoder]: Expect ids dimension [n] or [1,n].", ORT_INVALID_GRAPH);
  }

  //  const int64_t* p_row_indices = ort_row_indices_dim.empty() ? nullptr : ort_.GetTensorData<int64_t>(ort_row_indices);
  const OrtValue* positions = ort_.KernelContext_GetInput(context, 1);
  OrtTensorDimensions positions_dim(ort_, positions);
  if (use_indices_ &&
      (!((positions_dim.Size() == 0) ||
         (positions_dim.size() == 2 && positions_dim[1] == 2)))) {
    ORTX_CXX_API_THROW("[BertTokenizerDecoder]: Expect positions empty or a [n, 2] matrix when use indices", ORT_INVALID_GRAPH);
  }

  const int64_t* p_positions = positions_dim.Size() == 0 ? nullptr : ort_.GetTensorData<int64_t>(positions);

  std::vector<std::string> result;
  std::vector<int64_t> output_dim(1);
  if (!use_indices_) {
    result.push_back(decoder_->Decode(std::vector<int64_t>(p_ids, p_ids + ids_dim.Size()), skip_special_tokens_, clean_up_tokenization_spaces_));
    output_dim[0] = 1;
  } else {
    if (p_positions != nullptr) {
      for (int i = 0; i < positions_dim[0]; i++) {
        int64_t start = p_positions[2 * i];
        int64_t end = p_positions[2 * i + 1];

        result.push_back(decoder_->Decode(std::vector<int64_t>(p_ids + start, p_ids + end), skip_special_tokens_, clean_up_tokenization_spaces_));
      }
      output_dim[0] = positions_dim[0];
    }
  }
  OrtValue* output = ort_.KernelContext_GetOutput(context, 0, output_dim.data(), output_dim.size());

  FillTensorDataString(api_, ort_, context, result, output);
}

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

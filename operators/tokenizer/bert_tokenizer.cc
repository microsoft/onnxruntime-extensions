#include "bert_tokenizer.hpp"

#include <utility>

WordpieceTokenizer::WordpieceTokenizer(std::shared_ptr<std::unordered_map<ustring, int32_t>> vocab, ustring unk_token,
                                       ustring suffix_indicator, int max_input_chars_per_word): vocab_(std::move(vocab)), unk_token_(unk_token),
                                       suffix_indicator_(std::move(suffix_indicator)), max_input_chars_per_word_(max_input_chars_per_word) {
  auto it = vocab_->find(unk_token);
  if (it == vocab_->end()) {
    ORT_CXX_API_THROW("[WordpieceTokenizer]: can not find unk_token in vocal", ORT_RUNTIME_EXCEPTION);
  }
  unk_token_id_ = it->second;
}

std::vector<ustring> WordpieceTokenizer::Tokenize(const ustring& text) {
  std::vector<ustring> result;
  ustring token;
  for (auto c : text) {
    if (c == U' ' && !token.empty()) {
      GreedySearch(token, result);
      token.clear();
      continue;
    }

    token.push_back(c);
  }

  if (!token.empty()) {
    GreedySearch(token, result);
  }

  return result;
}

std::vector<ustring> WordpieceTokenizer::Tokenize(const std::vector<ustring>& tokens) {
  std::vector<ustring> result;
  for (const auto& token : tokens) {
    GreedySearch(token, result);
  }

  return result;
}

std::vector<int64_t> WordpieceTokenizer::Encode(const std::vector<ustring>& tokens) {
  std::vector<int64_t> ids;
  for (const auto& token : tokens) {
    auto it = vocab_->find(token);
    if (it == vocab_->end()) {
      ids.push_back(unk_token_id_);
      continue;
    }

    ids.push_back(it->second);
  }
  return ids;
}

void WordpieceTokenizer::GreedySearch(const ustring& token, std::vector<ustring>& tokenized_result) {
  if (token.size() > max_input_chars_per_word_) {
    tokenized_result.push_back(unk_token_);
    return;
  }

  int start = 0;
  int end = -1;
  ustring substr;
  for (; start < token.size();) {
    end = token.size();
    bool is_found = false;
    // try to found longest matched sub-token in vocab
    for (; start < end;) {
      substr = static_cast<const ustring>(token.substr(start, end - start));
      if (start > 0) {
        substr = static_cast<const ustring>(suffix_indicator_ + substr);
      }
      auto it = vocab_->find(substr);
      if (it != vocab_->end()) {
        is_found = true;
        break;
      }
      end -= 1;
    }
    // token not found in vocab
    if (!is_found) {
      tokenized_result.push_back(unk_token_);
      break;
    }

    tokenized_result.push_back(substr);
    start = end;
  }
}

BertTokenizer::BertTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize, ustring unk_token, ustring sep_token,
                             ustring pad_token, ustring cls_token, ustring mask_token, bool tokenize_chinese_chars, bool strip_accents,
                             ustring suffix_indicator) : do_basic_tokenize_(do_basic_tokenize) {
  auto tokens = SplitString(vocab, "\n", true);

  vocab_ = std::make_shared<std::unordered_map<ustring, int32_t>>();
  for (int i = 0; i < tokens.size(); i++) {
    (*vocab_)[ustring(tokens[i])] = i;
  }

  if (do_basic_tokenize) {
    basic_tokenizer_ = std::make_shared<BasicTokenizer>(do_lower_case, tokenize_chinese_chars, strip_accents, true, true);
  }
  wordpiece_tokenizer_ = std::make_shared<WordpieceTokenizer>(vocab_, unk_token, suffix_indicator);

  unk_token_id_ = FindSpecialToken(unk_token);
  sep_token_id_ = FindSpecialToken(sep_token);
  pad_token_id_ = FindSpecialToken(pad_token);
  cls_token_id_ = FindSpecialToken(cls_token);
  mask_token_id_ = FindSpecialToken(mask_token);
}
std::vector<ustring> BertTokenizer::Tokenize(const ustring& text) {
  if (do_basic_tokenize_) {
    return wordpiece_tokenizer_->Tokenize(basic_tokenizer_->Tokenize(text));
  }
  return wordpiece_tokenizer_->Tokenize(text);
}

std::vector<int64_t> BertTokenizer::Encode(const std::vector<ustring>& tokens) {
  return wordpiece_tokenizer_->Encode(tokens);
}

std::vector<int64_t> BertTokenizer::AddSpecialToken(const std::vector<int64_t>& ids) {
  std::vector<int64_t> result;
  result.reserve(ids.size() + 2);
  result.push_back(cls_token_id_);
  result.insert(result.end(), ids.begin(), ids.end());
  result.push_back(sep_token_id_);
  return result;
}

std::vector<int64_t> BertTokenizer::AddSpecialToken(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2) {
  std::vector<int64_t> result;
  result.reserve(ids1.size() + ids2.size() + 3);
  result.push_back(cls_token_id_);
  result.insert(result.end(), ids1.begin(), ids1.end());
  result.push_back(sep_token_id_);
  result.insert(result.end(), ids2.begin(), ids2.end());
  result.push_back(sep_token_id_);
  return result;
}

std::vector<int64_t> BertTokenizer::GenerateTypeId(const std::vector<int64_t>& ids) {
  return std::vector<int64_t>(ids.size() + 2, 0);
}

std::vector<int64_t> BertTokenizer::GenerateTypeId(const std::vector<int64_t>& ids1, const std::vector<int64_t>& ids2) {
  std::vector<int64_t> result;
  result.reserve(ids1.size() + ids2.size() + 3);
  result.insert(result.end(),  ids1.size() + 2, 0);
  result.insert(result.end(),  ids2.size() + 1, 1);
  return result;
}

int32_t BertTokenizer::FindSpecialToken(ustring token) {
  auto it = vocab_->find(token);
  if (it == vocab_->end()) {
    ORT_CXX_API_THROW("[BertTokenizer]: can not find special tokens: " + std::string(token), ORT_RUNTIME_EXCEPTION);
  }
  return it->second;
}

TruncateStrategy::TruncateStrategy(std::string strategy) {
  if (strategy == "longest_first") {
    strategy_ = TruncateStrategyType::LONGEST_FIRST;
  } else if (strategy == "only_first") {
    strategy_ = TruncateStrategyType::ONLY_FIRST;
  } else if (strategy == "only_second") {
    strategy_ = TruncateStrategyType::ONLY_SECOND;
  } else if (strategy == "longest_from_back") {
    strategy_ = TruncateStrategyType::LONGEST_FROM_BACK;
  }
}

void TruncateStrategy::Truncate(std::vector<int64_t>& ids, int64_t max_len) {
  if (max_len < 0) {
    return;
  }

  ids.resize(max_len);
}

void TruncateStrategy::Truncate(std::vector<int64_t>& input1, std::vector<int64_t>& input2, int64_t max_len) {

  if (max_len < 0 || (input1.size() + input2.size() <= max_len)) {
    return;
  }

  auto input1_keep_len = input1.size();
  auto input2_keep_len = input2.size();
  auto half_max_len = max_len / 2;

  switch (strategy_) {
    case TruncateStrategyType::LONGEST_FIRST:
    case TruncateStrategyType::LONGEST_FROM_BACK:

      if ((input1_keep_len > half_max_len) && (input2_keep_len > half_max_len)) {
        input1_keep_len = half_max_len;
        input2_keep_len = half_max_len;
      } else if (input2_keep_len > input1_keep_len) {
        input2_keep_len = max_len - input1_keep_len;
      } else {
        input1_keep_len = max_len - input2_keep_len;
      }

      if (strategy_ == TruncateStrategyType::LONGEST_FIRST) {
        input1.resize(input1_keep_len);
        input2.resize(input2_keep_len);
      } else {
        input1.erase(input1.begin(), input1.end() - input1_keep_len);
        input2.erase(input2.begin(), input2.end() - input2_keep_len);
      }

      return;
    case TruncateStrategyType::ONLY_FIRST:
      return;
    case TruncateStrategyType::ONLY_SECOND:
      return;
    default:
      return;
  }
}

KernelBertTokenizer::KernelBertTokenizer(OrtApi api, const OrtKernelInfo* info) : BaseKernel(api, info) {
  std::string vocab = ort_.KernelInfoGetAttribute<std::string>(info, "vocab_file");
  bool do_lower_case = TryToGetAttributeWithDefault("do_lower_case", true);
  bool do_basic_tokenize = TryToGetAttributeWithDefault("do_basic_tokenize", true);
  std::string unk_token = TryToGetAttributeWithDefault("unk_token", std::string("[UNK]"));
  std::string sep_token = TryToGetAttributeWithDefault("sep_token", std::string("[SEP]"));
  std::string pad_token = TryToGetAttributeWithDefault("pad_token", std::string("[PAD]"));
  std::string cls_token = TryToGetAttributeWithDefault("cls_token", std::string("[CLS]"));
  std::string mask_token = TryToGetAttributeWithDefault("mask_token", std::string("[MASK]"));
  bool tokenize_chinese_chars = TryToGetAttributeWithDefault("tokenize_chinese_chars", true);
  bool strip_accents = TryToGetAttributeWithDefault("strip_accents", false);
  std::string suffix_indicator = TryToGetAttributeWithDefault("suffix_indicator", std::string("##"));

  tokenizer_ = std::make_shared<BertTokenizer>(vocab, do_lower_case, do_basic_tokenize, ustring(unk_token),
                                               ustring(sep_token), ustring(pad_token),ustring(cls_token),
                                               ustring(mask_token), tokenize_chinese_chars, strip_accents, ustring(suffix_indicator));
}

void KernelBertTokenizer::Compute(OrtKernelContext* context) {
  // Setup inputs
  const OrtValue* input = ort_.KernelContext_GetInput(context, 0);
  std::vector<std::string> input_data;
  GetTensorMutableDataString(api_, ort_, context, input, input_data);

  if (input_data.size() != 1 && input_data.size() != 2) {
    ORT_CXX_API_THROW("[BertTokenizer]: only support one or two query.", ORT_INVALID_GRAPH);
  }
  std::vector<int64_t> input_ids;
  std::vector<int64_t> token_type_ids;

  if (input_data.size() == 1 || input_data[1].empty()) {
    std::vector<int64_t> encode = tokenizer_->Encode(tokenizer_->Tokenize(ustring(input_data[0])));
    input_ids = tokenizer_->AddSpecialToken(encode);
    token_type_ids = tokenizer_->GenerateTypeId(encode);
  } else {
    std::vector<int64_t> encode1 = tokenizer_->Encode(tokenizer_->Tokenize(ustring(input_data[0])));
    std::vector<int64_t> encode2 = tokenizer_->Encode(tokenizer_->Tokenize(ustring(input_data[1])));
    input_ids = tokenizer_->AddSpecialToken(encode1, encode2);
    token_type_ids = tokenizer_->GenerateTypeId(encode1, encode2);
  }

  std::vector<int64_t> attention_mask(input_ids.size(), 1);

  std::vector<int64_t> output_dim({static_cast<int64_t>(input_ids.size())});

  SetOutput(context, 0, output_dim, input_ids);
  SetOutput(context, 1, output_dim, token_type_ids);
  SetOutput(context, 2, output_dim, attention_mask);
}

void* CustomOpBertTokenizer::CreateKernel(OrtApi api, const OrtKernelInfo* info) const {
  return new KernelBertTokenizer(api, info);
};

const char* CustomOpBertTokenizer::GetName() const { return "BertTokenizer"; };

size_t CustomOpBertTokenizer::GetInputTypeCount() const {
  return 1;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetInputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING;
};

size_t CustomOpBertTokenizer::GetOutputTypeCount() const {
  return 3;
};

ONNXTensorElementDataType CustomOpBertTokenizer::GetOutputType(size_t /*index*/) const {
  return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
};



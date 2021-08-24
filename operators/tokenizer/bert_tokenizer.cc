#include "nlohmann/json.hpp"
#include "bert_tokenizer.hpp"

WordpieceTokenizer::WordpieceTokenizer(std::shared_ptr<std::unordered_map<ustring, int32_t>> vocab, ustring unk_token,
                                       ustring suffix_indicator, int max_input_chars_per_word): vocab_(vocab), unk_token_(unk_token),
                                       suffix_indicator_(suffix_indicator), max_input_chars_per_word_(max_input_chars_per_word) {
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

void WordpieceTokenizer::GreedySearch(const ustring& token, std::vector<ustring> tokenized_result) {
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
  std::unordered_map<std::string, int32_t> vocab_map;
  auto parsed = nlohmann::json::parse(vocab);
  parsed.get_to(vocab_map);

  for (auto& it : vocab_map) {
    (*vocab_)[ustring(it.first)] = it.second;
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

int32_t BertTokenizer::FindSpecialToken(ustring token) {
  auto it = vocab_->find(token);
  if (it == vocab_->end()) {
    ORT_CXX_API_THROW("[BertTokenizer]: can not find special tokens: " + std::string(token), ORT_RUNTIME_EXCEPTION);
  }
  return it->second;
}


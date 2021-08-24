#include "nlohmann/json.hpp"
#include "bert_tokenizer.hpp"

WordpieceTokenizer::WordpieceTokenizer(const std::string& vocab, ustring unk_token, ustring suffix_indicator, int max_input_chars_per_word):
                                    unk_token_(unk_token), suffix_indicator_(suffix_indicator), max_input_chars_per_word_(max_input_chars_per_word) {
  std::unordered_map<std::string, int32_t> vocab_map;
  auto parsed = nlohmann::json::parse(vocab);
  parsed.get_to(vocab_map);

  for (auto& it : vocab_map) {
    vocab_[ustring(it.first)] = it.second;
  }
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

  for (const auto& token: tokens) {
    GreedySearch(token, result);
  }

  return result;
}

std::vector<int64_t> WordpieceTokenizer::Encode(const std::vector<ustring>& tokens) {
  return std::vector<int64_t>();
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
      auto it = vocab_.find(substr);
      if (it != vocab_.end()) {
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
                             ustring suffix_indicator, int64_t max_input_chars_per_word) : do_basic_tokenize_(do_basic_tokenize), unk_token_(unk_token), sep_token_(sep_token), pad_token_(pad_token), cls_token_(cls_token), mask_token_(mask_token){
  auto parsed = nlohmann::json::parse(vocab);
  parsed.get_to(vocab_);
  if (do_basic_tokenize) {
    basic_tokenizer_ = std::make_shared<BasicTokenizer>(do_lower_case, tokenize_chinese_chars, strip_accents, true, true);
  }
  wordpiece_tokenizer_ = std::make_shared<WordpieceTokenizer>(vocab, unk_token, suffix_indicator);
}
std::vector<ustring> BertTokenizer::Tokenize(const ustring& text) {
  std::vector<ustring> result;
  if (do_basic_tokenize_) {
     wordpiece_tokenizer_->Tokenize(basic_tokenizer_->Tokenize(text));
  }

  result.insert(result.begin(), cls_token_);


}

std::vector<int64_t> BertTokenizer::Encode(const std::vector<ustring>& tokens) {
  return std::vector<int64_t>();
}

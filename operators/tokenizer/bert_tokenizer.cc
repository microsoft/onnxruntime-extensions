#include "nlohmann/json.hpp"
#include "bert_tokenizer.hpp"

BertTokenizer::BertTokenizer(std::string vocab, bool do_lower_case, bool do_basic_tokenize,ustring unk_token,ustring sep_token,
                                       ustring pad_token,ustring cls_token,ustring mask_token,bool tokenize_chinese_chars, bool strip_accents,
                                       ustring suffix_indicator,int64_t max_input_chars_per_word): do_basic_tokenize_(do_basic_tokenize),unk_token_(unk_token),
                                                                                                   sep_token_(sep_token),pad_token_(pad_token), cls_token_(cls_token),
                                                                                                   mask_token_(mask_token), suffix_indicator_(suffix_indicator){
  auto parsed = nlohmann::json::parse(vocab);
  parsed.get_to(vocab_);
  if (do_basic_tokenize) {
    basic_tokenizer_ = std::make_shared<BasicTokenizer>(do_lower_case,  tokenize_chinese_chars, strip_accents, true, true);
  }
}
std::vector<int64_t> BertTokenizer::Tokenize(std::string) {
  return std::vector<int64_t>();
}
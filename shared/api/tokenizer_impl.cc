// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "bpe_kernels.h"
#include "bpe_tokenizer.hpp"
#include "bpe_decoder.hpp"

#include "tokenizer_impl.h"

using namespace ort_extensions;

TokenizerImpl::TokenizerImpl() : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenizer) {};
TokenizerImpl::~TokenizerImpl() {};

OrtxStatus TokenizerImpl::Load(const std::string& dir) {
  tok_config_ = std::make_shared<ort_extensions::bpe::TokenJsonConfig>();
  auto status = tok_config_->Load(dir);
  if (!status.IsOk()) {
    return status;
  }

  tokenizer_ = std::make_unique<JsonFastTokenizer>();
  // load the tokenizer from a config
  status = tokenizer_->Load(*tok_config_);
  if (status.IsOk()) {
    detokenizer_ = std::make_unique<BpeStreamingDecoder>();
    status = detokenizer_->Load(tok_config_, *tokenizer_);
  }

  return status;
}

OrtxStatus TokenizerImpl::BatchEncode(const std::vector<std::string_view>& input,
                                      std::vector<std::vector<extTokenId_t>>& t_ids) const {
  for (const auto& s : input) {
    ortc::Tensor<int64_t> ts_output(&CppAllocator::Instance());
    ortc::Tensor<std::string> ts_input = ortc::Tensor<std::string>(std::vector<std::string>{std::string(s)});
    auto status = tokenizer_->Compute(ts_input, ts_output, std::nullopt, std::nullopt);

    if (!status.IsOk()) {
      return status;
    }

    std::vector<extTokenId_t> ids(ts_output.NumberOfElement());
    std::transform(ts_output.Data(), ts_output.Data() + ts_output.NumberOfElement(), ids.begin(),
                   [](int64_t v) { return static_cast<extTokenId_t>(v); });
    t_ids.emplace_back(std::move(ids));
  }

  return {};
}

OrtxStatus TokenizerImpl::BatchDecode(const std::vector<span<extTokenId_t const>>& t_ids,
                                      std::vector<std::string>& t_text) const {
  for (const auto& s : t_ids) {
    std::vector<int64_t> ids(s.size());
    std::transform(s.begin(), s.end(), ids.begin(), [](extTokenId_t v) { return static_cast<int64_t>(v); });
    ortc::Tensor<int64_t> ts_input(std::vector<int64_t>{1, static_cast<int64_t>(ids.size())}, (void*)ids.data());
    ortc::Tensor<std::string> ts_output;
    OrtxStatus status = detokenizer_->Compute(ts_input, ts_output);
    if (!status.IsOk()) {
      return status;
    }
    t_text.emplace_back(ts_output.AsScalar());
  }
  return {};
}

OrtxStatus TokenizerImpl::Id2Token(extTokenId_t id, std::string& token, BPEDecoderState** state) const {
  return detokenizer_->Id2Token(id, token, state);
}

static std::map<std::string, std::string> LANGUAGES = {
    {"en", "english"},        {"zh", "chinese"},       {"de", "german"},    {"es", "spanish"},    {"ru", "russian"},
    {"ko", "korean"},         {"fr", "french"},        {"ja", "japanese"},  {"pt", "portuguese"}, {"tr", "turkish"},
    {"pl", "polish"},         {"ca", "catalan"},       {"nl", "dutch"},     {"ar", "arabic"},     {"sv", "swedish"},
    {"it", "italian"},        {"id", "indonesian"},    {"hi", "hindi"},     {"fi", "finnish"},    {"vi", "vietnamese"},
    {"he", "hebrew"},         {"uk", "ukrainian"},     {"el", "greek"},     {"ms", "malay"},      {"cs", "czech"},
    {"ro", "romanian"},       {"da", "danish"},        {"hu", "hungarian"}, {"ta", "tamil"},      {"no", "norwegian"},
    {"th", "thai"},           {"ur", "urdu"},          {"hr", "croatian"},  {"bg", "bulgarian"},  {"lt", "lithuanian"},
    {"la", "latin"},          {"mi", "maori"},         {"ml", "malayalam"}, {"cy", "welsh"},      {"sk", "slovak"},
    {"te", "telugu"},         {"fa", "persian"},       {"lv", "latvian"},   {"bn", "bengali"},    {"sr", "serbian"},
    {"az", "azerbaijani"},    {"sl", "slovenian"},     {"kn", "kannada"},   {"et", "estonian"},   {"mk", "macedonian"},
    {"br", "breton"},         {"eu", "basque"},        {"is", "icelandic"}, {"hy", "armenian"},   {"ne", "nepali"},
    {"mn", "mongolian"},      {"bs", "bosnian"},       {"kk", "kazakh"},    {"sq", "albanian"},   {"sw", "swahili"},
    {"gl", "galician"},       {"mr", "marathi"},       {"pa", "punjabi"},   {"si", "sinhala"},    {"km", "khmer"},
    {"sn", "shona"},          {"yo", "yoruba"},        {"so", "somali"},    {"af", "afrikaans"},  {"oc", "occitan"},
    {"ka", "georgian"},       {"be", "belarusian"},    {"tg", "tajik"},     {"sd", "sindhi"},     {"gu", "gujarati"},
    {"am", "amharic"},        {"yi", "yiddish"},       {"lo", "lao"},       {"uz", "uzbek"},      {"fo", "faroese"},
    {"ht", "haitian creole"}, {"ps", "pashto"},        {"tk", "turkmen"},   {"nn", "nynorsk"},    {"mt", "maltese"},
    {"sa", "sanskrit"},       {"lb", "luxembourgish"}, {"my", "myanmar"},   {"bo", "tibetan"},    {"tl", "tagalog"},
    {"mg", "malagasy"},       {"as", "assamese"},      {"tt", "tatar"},     {"haw", "hawaiian"},  {"ln", "lingala"},
    {"ha", "hausa"},          {"ba", "bashkir"},       {"jw", "javanese"},  {"su", "sundanese"},  {"yue", "cantonese"}};

OrtxStatus TokenizerImpl::GetDecoderPromptIds(size_t batch_size, const char* lang, const char* task, int no_timestamps,
                                              std::vector<std::vector<extTokenId_t>>& t_ids) const {
  if (tokenizer_ == nullptr) {
    return OrtxStatus(kOrtxErrorInvalidArgument, "Tokenizer is not loaded");
  }
  // since it was only supported by Whisper model, should we check it here?

  auto convert_tokens_to_ids = [this](const std::string& token) -> extTokenId_t {
    ortc::Tensor<int64_t> ts_output(&CppAllocator::Instance());
    ortc::Tensor<std::string> ts_input = ortc::Tensor<std::string>(std::vector<std::string>{std::string(token)});
    auto status = this->tokenizer_->Compute(ts_input, ts_output, std::nullopt, std::nullopt);
    if (!status.IsOk()) {
      return static_cast<extTokenId_t>(-1);
    }
    return static_cast<extTokenId_t>(ts_output.Data()[0]);
  };

  auto translate_token_id = convert_tokens_to_ids("<|translate|>");
  auto transcribe_token_id = convert_tokens_to_ids("<|transcribe|>");
  auto notimestamps_token_id = convert_tokens_to_ids("<|notimestamps|>");
  std::vector<extTokenId_t> ids;
  ids.reserve(4);
  if (lang != nullptr) {
    auto lang_str = LANGUAGES.find(lang);
    if (lang_str == LANGUAGES.end()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Invalid language");
    }
    ids.push_back(convert_tokens_to_ids(lang_str->second));
  }

  if (task != nullptr) {
    if (!strcmp(task, "translation") || !strcmp(task, "translate")) {
      ids.push_back(translate_token_id);
    } else if (!strcmp(task, "transcription") || !strcmp(task, "transcribe")) {
      ids.push_back(transcribe_token_id);
    } else {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Invalid task");
    }
  }

  if (no_timestamps) {
    ids.push_back(notimestamps_token_id);
  }

  t_ids.resize(batch_size, ids);
  return {};
}

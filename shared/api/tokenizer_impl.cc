// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bpe_kernels.h"
#include "bpe_tokenizer_model.hpp"
#include "bpe_decoder.hpp"
#include "ugm_kernels.hpp"

#include "tokenizer_impl.h"


namespace ort_extensions {

std::set<std::string> TokenizerImpl::supported_bpe_models_ = {
  "PreTrainedTokenizerFast", 
  "CLIPTokenizer",
  "WhisperTokenizer",
  "GemmaTokenizer",
  "LlamaTokenizer",
  "Phi3Tokenizer",
  "CodeLlamaTokenizer",
  "CodeGenTokenizer",
  "GPT2Tokenizer",
  "Qwen2Tokenizer"
};

std::set<std::string> TokenizerImpl::supported_ugm_models_ = {
  "XLMRobertaTokenizer"
};

TokenizerImpl::TokenizerImpl()
    : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenizer) {};
TokenizerImpl::~TokenizerImpl() {};

OrtxStatus TokenizerImpl::LoadTokenizer(const OrtxTokenizerBlob* blob) {
  if (tok_config_->tokenizer_class_.empty() ||
      supported_ugm_models_.count(tok_config_->tokenizer_class_)) {
    auto tokenizer = std::make_unique<SpmUgmTokenizer>();
    auto status = tokenizer->Load(*tok_config_);
    if (!status.IsOk()) {
      return status;
    }
    auto detok = std::make_unique<SpmUgmDecoder>();

    if (status.IsOk()) {
      status = detok->Load(*tok_config_, *tokenizer);
    }

    if (status.IsOk()) {
      tokenizer_ = std::move(tokenizer);
      detokenizer_ = std::move(detok);
    }

    return status;
  }

  if (!supported_bpe_models_.count(tok_config_->tokenizer_class_)) {
    return OrtxStatus(kOrtxErrorNotImplemented, "Unsupported tokenizer class");
  }

  auto tokenizer = std::make_unique<JsonFastTokenizer>();
  auto fx_load = &JsonFastTokenizer::Load;
  if (blob == nullptr) {
    auto vocab_file_path = ortx::path(tok_config_->GetVocabDataFile());
    // vocab file is checked in TokenJsonConfig::Load
    if (vocab_file_path.extension() != ".json") {
      fx_load = &JsonFastTokenizer::LoadTikTokenBase64;
    }
  } else {
    if (blob->raw_model_blob_len > 0) {
      fx_load = &JsonFastTokenizer::LoadTikTokenBase64;
    }
  }

  auto status = (tokenizer.get()->*fx_load)(*tok_config_);
  if (!status.IsOk()) {
    return status;
  }

  auto detok = std::make_unique<BpeStreamingDecoder>();
  status = detok->Load(tok_config_, *tokenizer);

  if (status.IsOk()) {
    tokenizer_ = std::move(tokenizer);
    detokenizer_ = std::move(detok);
  }

  return status;
}

OrtxStatus TokenizerImpl::Load(const OrtxTokenizerBlob& blob) {
  tok_config_ = std::make_shared<ort_extensions::TokenJsonConfig>();
  auto status = tok_config_->LoadFromBlob(blob);
  if (!status.IsOk()) {
    return status;
  }

  return LoadTokenizer(&blob);
}

OrtxStatus TokenizerImpl::Load(const std::string& tok_path) {
  tok_config_ = std::make_shared<ort_extensions::TokenJsonConfig>();
  auto status = tok_config_->Load(tok_path);
  if (!status.IsOk()) {
    return status;
  }

  return LoadTokenizer();
}

OrtxStatus TokenizerImpl::BatchEncode(const std::vector<std::string_view>& input,
                                      std::vector<std::vector<extTokenId_t>>& t_ids) const {
  for (const auto& s : input) {
    ortc::Tensor<int64_t> ts_output(&CppAllocator::Instance());
    ortc::Tensor<std::string> ts_input = ortc::Tensor<std::string>(std::vector<std::string>{std::string(s)});
    
    OrtxStatus status = std::visit([&](auto& tokenizer) {
      return tokenizer->Compute(ts_input, ts_output);
    }, tokenizer_);

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
    OrtxStatus status = std::visit([&](auto& detokenizer) {
      return detokenizer->Compute(ts_input, ts_output); }, detokenizer_);

    if (!status.IsOk()) {
      return status;
    }
    t_text.push_back(ts_output.AsScalar());
  }
  return {};
}

OrtxStatus TokenizerImpl::Id2Token(extTokenId_t id, std::string& token, TokenizerDecodingState** state) const {
  return std::visit([&](auto& detokenizer) {
    return detokenizer->Id2Token(id, token, state); }, detokenizer_);
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
    {"te", "telugu"},         {"fa", "persian"},       {"lv", "latvian"},   {"bn", "bangla"},     {"sr", "serbian"},
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
  // since it was only supported by Whisper model, which is bpe only.
  if (!std::holds_alternative<bpe_tokenizer_t>(tokenizer_)) {
    return OrtxStatus(kOrtxErrorInvalidArgument, "Tokenizer is not loaded");
  }

  auto translate_token_id = std::get<bpe_tokenizer_t>(tokenizer_)->GetTokenId("<|translate|>");
  auto transcribe_token_id = std::get<bpe_tokenizer_t>(tokenizer_)->GetTokenId("<|transcribe|>");
  auto notimestamps_token_id = std::get<bpe_tokenizer_t>(tokenizer_)->GetTokenId("<|notimestamps|>");
  std::vector<extTokenId_t> ids;
  ids.reserve(4);
  if (lang != nullptr) {
    auto lang_str = LANGUAGES.find(lang);
    if (lang_str == LANGUAGES.end()) {
      return OrtxStatus(kOrtxErrorInvalidArgument, "Invalid language");
    }

    std::string lang_token = "<|" + lang_str->first + "|>";
    ids.push_back(std::get<bpe_tokenizer_t>(tokenizer_)->GetTokenId(lang_token));
  }

  if (task != nullptr) {
    if (0 == strcmp(task, "translate") == 0) {
      ids.push_back(translate_token_id);
    } else if (0 == strcmp(task, "transcribe")) {
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

}  // namespace ort_extensions

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "bpe_kernels.h"
#include "bpe_tokenizer_model.hpp"
#include "bpe_decoder.hpp"
#include "ugm_kernels.hpp"

#include "tokenizer_impl.h"


namespace ort_extensions {


TokenizerImpl::TokenizerImpl()
    : OrtxObjectImpl(extObjectKind_t::kOrtxKindTokenizer) {};
TokenizerImpl::~TokenizerImpl() {};

OrtxStatus TokenizerImpl::LoadTokenizer(const OrtxTokenizerBlob* blob) {

  auto type = TokenJsonConfig::GetTokenType(tok_config_->tokenizer_class_);
  if (type == TokenType::kUnigram) {
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
  } else if (type == TokenType::kBPE) {
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

  return OrtxStatus(kOrtxErrorNotImplemented, "Unsupported tokenizer class: " + tok_config_->tokenizer_class_);
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

  chat_template = tok_config_->chat_template_;
  
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

// Constant string variable to store predefined chat template strings for popular supported models
const std::string PHI4_CHAT_TEMPLATE = R"({% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %})";
const std::string PHI3_5_CHAT_TEMPLATE = R"({% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %})";
const std::string LLAMA3_CHAT_TEMPLATE = R"({{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n)";

// Member variable to store the messages
std::vector<std::unordered_map<std::string, std::string>> messages;
    
// Member variable to store the chat_template (customized for each instance)
std::string chat_template;

// Phi4ChatTemplate method to process messages and store result in output
OrtxStatus TokenizerImpl::Phi4ChatTemplate(std::string* output, bool add_generation_prompt = true, const std::string& eos_token = "<|endoftext|>") {
    // Clear the output string before starting
    output->clear();

    // Process the messages
    for (const auto& message : messages) {
        std::string role = message.at("role");
        std::string content = message.at("content");

        // Check if "tools" is present in the message and is not empty for "system" role
        if (role == "system" && message.find("tools") != message.end() && !message.at("tools").empty()) {
            std::string tools = message.at("tools");
            *output += "<|" + role + "|>";
            *output += content + "<|tool|>" + tools + "<|/tool|>" + "<|end|>";
        } else {
            // For other messages, no tools
            *output += "<|" + role + "|>";
            *output += content + "<|end|>";
        }
    }

    // Add generation prompt or eos_token
    if (add_generation_prompt) {
        *output += "<|assistant|>";
    } else {
        *output += eos_token;
    }

    return OrtxStatus(kOrtxOK, "Created chat template.");
}

OrtxStatus TokenizerImpl::Phi3_5ChatTemplate(std::string* output, bool add_generation_prompt = true, const std::string& eos_token = "<|endoftext|>") {
  // Clear the output string before starting
  output->clear();

  // Process the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Check for different roles and format accordingly
      if (role == "system" && !content.empty()) {
          *output += "<|system|>\n";
          *output += content + "<|end|>\n";
      } else if (role == "user") {
          *output += "<|user|>\n";
          *output += content + "<|end|>\n";
      } else if (role == "assistant") {
          *output += "<|assistant|>\n";
          *output += content + "<|end|>\n";
      }
  }

  // Add generation prompt or eos_token
  if (add_generation_prompt) {
      *output += "<|assistant|>\n";
  } else {
      *output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created chat template.");
}

OrtxStatus TokenizerImpl::Llama3ChatTemplate(
  std::string* output, 
  bool add_generation_prompt = true, 
  const std::string& eos_token = "<|eot_id|>", 
  const std::vector<std::string>& custom_tools = {}, 
  bool tools_in_user_message = true, 
  const std::string& strftime_now = "",
  const std::string& bos_token = "<|begin_of_text|>") {  // Add bos_token as a parameter

  // Clear the output string before starting
  output->clear();

  // Prepend BOS token at the start of the output
  *output += bos_token + "\n";  // BOS token goes first

  // Initialize date_string with default value
  std::string date_string = "26 Jul 2024";  // Default date
  if (!strftime_now.empty()) {
      date_string = strftime_now;  // Override with provided date string if available
  }

  // Loop through messages and process each one
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Handle the system message
      if (role == "system") {
          *output += "<|start_header_id|>system<|end_header_id|>\n\n";
          *output += "Cutting Knowledge Date: December 2023\n";
          *output += "Today Date: " + date_string + "\n\n";

          // Check if tools exist and append relevant information
          if (!custom_tools.empty()) {
              *output += "You have access to the following functions. To call a function, please respond with JSON for a function call.\n";
              *output += "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value.}\n";
              *output += "Do not use variables.\n\n";

              // Convert tools to JSON (assuming custom_tools is a vector of tool names as strings)
              nlohmann::json tools_json = nlohmann::json::array();
              for (const auto& tool : custom_tools) {
                  tools_json.push_back(tool);
              }

              *output += tools_json.dump(4) + "\n\n";
          }
          *output += "<|eot_id|>\n";
      }

      // Handle user message with tools in it
      if (tools_in_user_message && message.find("tool_calls") != message.end()) {
          // Parse the tool_calls string into JSON (assuming it's a valid JSON string)
          nlohmann::json tool_calls_json = nlohmann::json::parse(message.at("tool_calls"));

          if (tool_calls_json.size() != 1) {
              // Handle multiple tool calls (not supported)
              return OrtxStatus(kOrtxErrorInvalidArgument, "This model only supports single tool-calls at once!");
          }

          // Extract the function name and arguments from the first tool call
          std::string function_name = tool_calls_json[0]["function"];
          nlohmann::json arguments = tool_calls_json[0]["arguments"];

          // Create the JSON object for the tool call
          nlohmann::json tool_call_json;
          tool_call_json["name"] = function_name;
          tool_call_json["parameters"] = arguments;

          // Serialize the tool call as JSON and append it to output
          *output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
          *output += tool_call_json.dump() + "\n";
          *output += "<|eot_id|>\n";  // End of tool call
      }

      // Handle other messages (user, assistant, etc.)
      else {
          *output += "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
          *output += content + "\n";
          *output += "<|eot_id|>\n";
      }
  }

  // Add generation prompt or eos_token at the end
  if (add_generation_prompt) {
      *output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  } else {
      *output += eos_token;  // Add the EOS token instead
  }

  return OrtxStatus(kOrtxOK, "Created chat template.");
}

// ApplyChatTemplate method to choose the template logic based on chat_template
OrtxStatus TokenizerImpl::ApplyChatTemplate(std::vector<std::unordered_map<std::string, std::string>> message_list, std::string* output, bool add_generation_prompt = true) {
    
    // Initialize messages
    messages = message_list;
  
    // Check if the chat_template matches any of the supported template strings and if so apply the corresponding template.
    if (chat_template == PHI4_CHAT_TEMPLATE) {
        return Phi4ChatTemplate(output, add_generation_prompt);
    } else if (chat_template == PHI3_5_CHAT_TEMPLATE) {
      return Phi3_5ChatTemplate(output, add_generation_prompt);
    } else if (chat_template == LLAMA3_CHAT_TEMPLATE) {
      return Llama3ChatTemplate(output, add_generation_prompt);
    } else {
        // Handle other templates or custom logic here
        return OrtxStatus(kOrtxErrorNotImplemented, "The provided chat template is currently not supported. Custom template handling needed.");
    }
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

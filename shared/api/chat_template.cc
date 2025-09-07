// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_impl.h"
namespace ort_extensions {

OrtxStatus TokenizerImpl::LoadChatTemplate() {
  // Load the chat template from the tokenizer configuration
  chat_template = (tok_config_->tokenizer_class_ == "WhisperTokenizer") ? R"({{ messages | map(attribute='content') | join('\n') }})" : tok_config_->chat_template_;
  if (chat_template.size()) {
    try {
      chat_template_root_ = minja::Parser::parse(chat_template, {});
    } catch (const std::runtime_error&) {
      return OrtxStatus(kOrtxOK, "Warning: The chat template for this model is not yet supported, trying to apply chat template will cause an error.");
    }
  }

  return OrtxStatus(kOrtxOK, "Loaded chat template.");
}

TokenizerImpl::MessageList TokenizerImpl::ParseJson(const std::string& json_str) {
  nlohmann::ordered_json json_obj = nlohmann::json::parse(json_str, nullptr, false, true);
  // Check if the parsed JSON is an array
  if (!json_obj.is_array()) {
    return {};
  }
  // Create a vector to hold the parsed messages
  MessageList messages;
  for (const auto& message : json_obj) {
    // Check if the message is an object
    if (!message.is_object()) {
      return {};
    }
    // Add the message to the vector
    auto msg = message.get<std::unordered_map<std::string, nlohmann::ordered_json>>();
    // convert msg to a string-string map
    std::unordered_map<std::string, std::string> msg_str;
    for (const auto& [key, value] : msg) {
      std::string value_str = value.dump();
      // remove the quotes from the string
      if (value_str.size() > 1 && value_str[0] == '"' && value_str[value_str.size() - 1] == '"') {
        value_str.erase(0, 1);
        value_str.erase(value_str.size() - 1, 1);
      }
      msg_str[key] = value_str;
    }
    messages.push_back(msg_str);
  }

  return messages;
}

void TokenizerImpl::InitializeChatParameters(const char* template_str,
                                             const std::vector<std::string>& custom_tools_param,
                                             bool tools_in_user_message_param, const std::string& strftime_param,
                                             const std::string& date_str,
                                             const std::vector<std::string>& builtin_tools_param) {
  // Initialize parameters with provided or default values
  custom_tools = custom_tools_param;
  tools_in_user_message = tools_in_user_message_param;
  strftime_now = strftime_param;
  date_string = date_str;
  builtin_tools = builtin_tools_param;

  bos_token = tok_config_->bos_token_;
  eos_token = tok_config_->eos_token_;
  if (template_str && *template_str) {
    chat_template = template_str;
  } else {
    chat_template = tok_config_->chat_template_;
  }
}

// This chat template implementation uses a Minja engine, a lightweight Jinja implementation in C++,
// hence it does not automatically support built-in filters or custom functions unless they are provided.
// We can thereby write common functions such as strftime_now here.

minja::Value strftime_function(const std::shared_ptr<minja::Context>&, minja::ArgumentsValue& args) {
    std::string format = "%Y-%m-%d";

    if (args.has_named("format") && args.get_named("format").is_string()) {
        format = args.get_named("format").to_str();
    } else if (!args.args.empty() && args.args[0].is_string()) {
        format = args.args[0].to_str();
    }

    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);

    std::tm tm;
#ifdef _WIN32
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif

    char buf[100];
    if (std::strftime(buf, sizeof(buf), format.c_str(), &tm)) {
        return minja::Value(std::string(buf));
    }
    return minja::Value("");
}

OrtxStatus TokenizerImpl::ApplyChatTemplate(const char* template_str, const char* message, const char* tools,
                                            std::string& output, std::vector<extTokenId_t>& ids_vec,
                                            bool add_generation_prompt, bool tokenize) const {
  OrtxStatus status;
  std::string input_str = minja::normalize_newlines(message);

  // Whisper does not have explicit chat template functionality.
  // However, the expected logic (same for HF and OAI) should emulate concatenating message['content'],
  // with no roles, separators, etc. We thereby automatically handle this in ORT Extensions as well.
  auto activated_str = (tok_config_->tokenizer_class_ == "WhisperTokenizer") ? R"({{ messages | map(attribute='content') | join('\n') }})" : tok_config_->chat_template_.c_str();
  
  if (template_str && *template_str) {
    activated_str = template_str;
  }

  if (*activated_str == '\0') {
    return {kOrtxErrorInvalidArgument, "Empty chat template."};
  }

  // Parse the chat template with Minja (a C++ Jinja templating engine).
  using json = nlohmann::ordered_json;
  std::string text;
  try {
    json actual_messages = json::parse(input_str.c_str());
    auto root = chat_template_root_;
    if (activated_str == template_str) {
      root = minja::Parser::parse(template_str, {});
    }
    if (root == nullptr) {
      // Note that this will get caught in the "catch" where we return status.
      throw std::runtime_error("Invalid or unsupported chat template.");
    }

    std::shared_ptr<minja::Context> context;

    if (tools && *tools) {
      std::string tools_str = minja::normalize_newlines(tools);
      json tools_json = json::parse(tools_str.c_str());
      context = minja::Context::make(json({
          {"messages", actual_messages},
          {"tools", tools_json},
          {"add_generation_prompt", add_generation_prompt},
      }));
    } else {
      context = minja::Context::make(json({
          {"messages", actual_messages},
          {"add_generation_prompt", add_generation_prompt},
      }));
    }

    context->set("strftime_now", minja::Value::callable(strftime_function));
    context->set("bos_token", tok_config_->bos_token_);
    context->set("eos_token", tok_config_->eos_token_);
    text = root->render(context);
    output = text;
  } catch (const std::runtime_error& e) {
    status = {kOrtxErrorInvalidArgument, e.what()};
  }

  if (!status.IsOk()) {
    return status;
  }

  if (tokenize) {
    status = std::visit([&](auto& tokenizer) { return tokenizer->ComputeNoOp(output, ids_vec, false); }, tokenizer_);
  }

  return status;
}

}  // namespace ort_extensions

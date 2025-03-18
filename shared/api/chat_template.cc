// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_impl.h"

namespace ort_extensions {


OrtxStatus TokenizerImpl::PhiVisionChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Iterate over the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Format the message according to the role
      output += "<|" + role + "|>\n" + content + "<|end|>\n";
  }

  // Check if a generation prompt is needed and the last message isn't from the assistant
  if (add_generation_prompt && messages.back().at("role") != "assistant") {
      output += "<|assistant|>\n";
  }

  return OrtxStatus(kOrtxOK, "Created Phi vision chat template.");
}

// Defaults: eos_token = "<|endoftext|>"
OrtxStatus TokenizerImpl::Phi3ChatTemplate(std::string& output, bool add_generation_prompt) const {
  // Clear the output string before starting
  output.clear();

  // Process the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Check for different roles and format accordingly
      if (role == "system" && !content.empty()) {
          output += "<|system|>\n";
          output += content + "<|end|>\n";
      } else if (role == "user") {
          output += "<|user|>\n";
          output += content + "<|end|>\n";
      } else if (role == "assistant") {
          output += "<|assistant|>\n";
          output += content + "<|end|>\n";
      }
  }

  // Add generation prompt or EOS token
  if (add_generation_prompt) {
      output += "<|assistant|>\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Phi-3 chat template.");
}

// Defaults: eos_token = "<|endoftext|>"
OrtxStatus TokenizerImpl::Phi3_5ChatTemplate(std::string& output, bool add_generation_prompt) const {
  // Clear the output string before starting
  output.clear();

  // Process the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Check for different roles and format accordingly
      if (role == "system") {
          output += "<|system|>\n";
          output += content + "<|end|>\n";
      } else if (role == "user") {
          output += "<|user|>\n";
          output += content + "<|end|>\n";
      } else if (role == "assistant") {
          output += "<|assistant|>\n";
          output += content + "<|end|>\n";
      }
  }

  // Add generation prompt or EOS token
  if (add_generation_prompt) {
      output += "<|assistant|>\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Phi-3.5 chat template.");
}

// Defaults: eos_token = "<|endoftext|>", bos_token = "<|startoftext|>"
OrtxStatus TokenizerImpl::Phi3SmallChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Add the beginning-of-sequence token
  output += bos_token;

  // Iterate over the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Format the message according to the role
      output += "<|" + role + "|>\n" + content + "<|end|>\n";
  }

  // Add the generation prompt or eos_token
  if (add_generation_prompt) {
      output += "<|assistant|>\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Phi-3-small chat template.");
}

OrtxStatus TokenizerImpl::Phi3MediumChatTemplate(std::string& output, bool add_generation_prompt) const {
  // Clear the output string before starting
  output.clear();

  // Process the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Format based on role (user/assistant)
      if (role == "user") {
          output += "<|user|>\n" + content + "<|end|>\n<|assistant|>\n";
      } else if (role == "assistant") {
          output += content + "<|end|>\n";
      }
  }

  return OrtxStatus(kOrtxOK, "Created Phi-3-medium chat template.");
}

// Defaults: eos_token = "<|endoftext|>"
OrtxStatus TokenizerImpl::Phi4ChatTemplate(std::string& output, bool add_generation_prompt) const {
  // Clear the output string before starting
  output.clear();

  // Process the messages
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Check if "tools" is present in the message and is not empty for "system" role
      if (role == "system" && message.find("tools") != message.end() && !message.at("tools").empty()) {
          std::string tools = message.at("tools");
          output += "<|" + role + "|>";
          output += content + "<|tool|>" + tools + "<|/tool|>" + "<|end|>";
      } else {
          // For other messages, no tools
          output += "<|" + role + "|>";
          output += content + "<|end|>";
      }
  }

  // Add generation prompt or eos_token
  if (add_generation_prompt) {
      output += "<|assistant|>";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Phi-4 chat template.");
}

// Defaults: eos_token = "</s>", bos_token = "<s>"
OrtxStatus TokenizerImpl::Llama2ChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Initialize system message and process it
  bool system_message_exists = false;
  std::string system_message = "";

  if (!messages.empty() && messages[0].at("role") == "system") {
      system_message = messages[0].at("content");
      system_message_exists = true;
  }

  // If system message exists, we start processing from the second message
  size_t start_index = system_message_exists ? 1 : 0;

  // Iterate over the messages to construct the template
  for (size_t i = start_index; i < messages.size(); ++i) {
      const auto& message = messages[i];
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Check if the conversation roles alternate between user and assistant
      if ((role == "user") != (i % 2 == start_index % 2)) {
          return OrtxStatus(kOrtxErrorInvalidArgument, "Conversation roles must alternate user/assistant/user/assistant...");
      }

      // Handle system message by prepending it to the first assistant's message
      std::string formatted_content;
      if (i == start_index && system_message_exists) {
          formatted_content = "<<SYS>>\n" + system_message + "\n<</SYS>>\n\n" + content;
      } else {
          formatted_content = content;
      }

      // Add the appropriate markers for user and assistant roles
      if (role == "user") {
          output += bos_token + "[INST] " + formatted_content + " [/INST]";
      } else if (role == "assistant") {
          output += " " + formatted_content + " " + eos_token;
      }
  }

  return OrtxStatus(kOrtxOK, "Created Llama 2 chat template.");
}

// Defaults: eos_token = "<|eot_id|>", bos_token = "<|begin_of_text|>"
OrtxStatus TokenizerImpl::Llama3ChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Iterate over the messages to construct the template
  for (size_t i = 0; i < messages.size(); ++i) {
      const auto& message = messages[i];
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Build the message with header and content
      std::string formatted_content = "<|start_header_id|>" + role + "<|end_header_id|>\n\n" + content + eos_token;

      // Add BOS token only to the first message
      if (i == 0) {
          formatted_content = bos_token + formatted_content;
      }

      // Append the formatted message to the output
      output += formatted_content;
  }

  // Add generation prompt or eos_token at the end
  if (add_generation_prompt) {
      output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Llama 3 chat template.");
}

// Defaults: eos_token = "<|eot_id|>", bos_token = "<|begin_of_text|>"
OrtxStatus TokenizerImpl::Llama3_2ChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Prepend BOS token at the start of the output
  output += bos_token;

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
          output += "<|start_header_id|>system<|end_header_id|>\n\n";
          output += "Cutting Knowledge Date: December 2023\n";
          output += "Today Date: " + date_string + "\n\n";

          // Check if tools exist and append relevant information
          if (!custom_tools.empty()) {
              output += "You have access to the following functions. To call a function, please respond with JSON for a function call.\n";
              output += "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value.}\n";
              output += "Do not use variables.\n\n";

              // Convert tools to JSON (assuming custom_tools is a vector of tool names as strings)
              nlohmann::json tools_json = nlohmann::json::array();
              for (const auto& tool : custom_tools) {
                  tools_json.push_back(tool);
              }

              output += tools_json.dump(4) + "\n\n";
          }
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
          output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
          output += tool_call_json.dump();
          output += eos_token;  // End of tool call
      }

      // Handle other messages (user, assistant, etc.)
      else {
          if (role != "system") {
            output += "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
          }
          output += content;
          output += eos_token;
      }
  }

  // Add generation prompt or eos_token at the end
  if (add_generation_prompt) {
      output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Llama 3.2 chat template.");
}

// Defaults: eos_token = "<|eot_id|>", bos_token = "<|begin_of_text|>", date_string = "26 Jul 2024"
OrtxStatus TokenizerImpl::Llama3_3ChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Prepend BOS token at the start of the output
  output += bos_token;

  // Loop through messages and process each one
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Handle the system message
      if (role == "system") {
          output += "<|start_header_id|>system<|end_header_id|>\n\n";
          output += "Cutting Knowledge Date: December 2023\n";
          output += "Today Date: " + date_string + "\n\n";

          // Check if builtin_tools or custom_tools exist and append relevant information
          if (!builtin_tools.empty() || !custom_tools.empty()) {
              output += "Environment: ipython\n";
          }

          // Add builtin tools if defined (excluding 'code_interpreter')
          if (!builtin_tools.empty()) {
              output += "Tools: ";
              bool first = true;
              for (const auto& tool : builtin_tools) {
                  if (tool != "code_interpreter") {
                      if (!first) {
                          output += ", ";
                      }
                      output += tool;
                      first = false;
                  }
              }
              output += "\n\n";
          }

          // Add the tools section if custom tools are provided
          if (!custom_tools.empty()) {
              output += "You have access to the following functions. To call a function, please respond with JSON for a function call.\n";
              output += "Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.\n";
              output += "Do not use variables.\n\n";

              // Convert custom_tools to JSON
              nlohmann::json tools_json = nlohmann::json::array();
              for (const auto& tool : custom_tools) {
                  tools_json.push_back(tool);
              }
              output += tools_json.dump(4) + "\n\n";
          }
      }

      // Handle user message with tools in it
      if (tools_in_user_message && message.find("tool_calls") != message.end()) {
          // Parse the tool_calls string into JSON
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

          // If the tool is a built-in tool, use the specific format for ipython
          bool is_builtin_tool = std::find(builtin_tools.begin(), builtin_tools.end(), function_name) != builtin_tools.end();
          if (is_builtin_tool) {
              output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
              output += "<|python_tag|>" + function_name + ".call(";
              bool first = true;
              for (auto& [arg_name, arg_val] : arguments.items()) {
                  if (!first) {
                      output += ", ";
                  }
                  output += arg_name + "=\"" + arg_val.get<std::string>() + "\"";
                  first = false;
              }
              output += ")";
          } else {
              output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
              output += tool_call_json.dump();
          }

          if (!builtin_tools.empty()) {
              output += "<|eom_id|>";
          } else {
              output += eos_token;
          }
      }

      // Handle other messages (user, assistant, etc.)
      else {
          if (role != "system") {
            output += "<|start_header_id|>" + role + "<|end_header_id|>\n\n";
          }
          output += content;
          output += eos_token;
      }
  }

  // Add generation prompt or eos_token at the end
  if (add_generation_prompt) {
      output += "<|start_header_id|>assistant<|end_header_id|>\n\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created Llama 3.1/3.3 chat template."); // Llama 3.1 and 3.3 have the same chat template
}

// Defaults: eos_token = "<｜end▁of▁sentence｜>", bos_token = "<｜begin▁of▁sentence｜>"
OrtxStatus TokenizerImpl::DeepSeekChatTemplate(std::string& output, bool add_generation_prompt) const {

  // Clear the output string before starting
  output.clear();

  // Initialize the namespace for template variables
  bool is_first = true;  // Track the first occurrence of the tool call or assistant message
  bool is_tool = false;
  bool is_output_first = true;
  std::string system_prompt = "";

  // Prepend BOS token at the start of the output
  output += bos_token;

  // Loop through messages and process each one
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Handle the system message
      if (role == "system") {
          system_prompt = content;
      }
  }

  output += system_prompt;  // Add system prompt to the output

  // Process each message in the conversation
  for (const auto& message : messages) {
      std::string role = message.at("role");
      std::string content = message.at("content");

      // Handle user message
      if (role == "user") {
          is_tool = false;
          output += "<｜User｜>" + content;
      }

      // Handle assistant message with tool calls
      if (role == "assistant" && message.find("tool_calls") != message.end()) {
          is_tool = false;

          // Parse the tool_calls string into JSON
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

          // Handle the first tool call differently
          if (is_first) {
              output += "<｜Assistant｜><｜tool_calls_begin｜><｜tool_call_begin｜>" + tool_calls_json[0]["type"].get<std::string>() + "<｜tool_sep｜>" + tool_calls_json[0]["function"]["name"].get<std::string>() + "\njson\n" + tool_calls_json[0]["function"]["arguments"].dump() + "\n<｜tool_call_end｜>";
              is_first = false;  // Mark as first tool call processed
          } else {
              // Subsequent tool calls
              output += "\n<｜tool_call_begin｜>" + tool_calls_json[0]["type"].get<std::string>() + "<｜tool_sep｜>" + tool_calls_json[0]["function"]["name"].get<std::string>() + "\njson\n" + tool_calls_json[0]["function"]["arguments"].dump() + "\n<｜tool_call_end｜>";
          }

          output += "<｜tool_calls_end｜>";
          output += eos_token;
      }

      // Handle assistant message without tool calls
      if (role == "assistant" && !content.empty()) {
          if (is_tool) {
              output += "<｜tool_outputs_end｜>" + content;
              output += eos_token;
              is_tool = false;
          } else {
              output += "<｜Assistant｜>" + content;
              output += eos_token;
          }
      }

      // Handle tool messages
      if (role == "tool") {
          is_tool = true;
          if (is_output_first) {
              output += "<｜tool_outputs_begin｜><｜tool_output_begin｜>" + content + "<｜tool_output_end｜>";
              is_output_first = false;
          } else {
              output += "\n<｜tool_output_begin｜>" + content + "<｜tool_output_end｜>";
          }
      }
  }

  // If still in a tool message, close it
  if (is_tool) {
      output += "<｜tool_outputs_end｜>";
  }

  // Add generation prompt or eos_token at the end
  if (add_generation_prompt && !is_tool) {
      output += "<｜Assistant｜><think>\n";
  } else {
      output += eos_token;
  }

  return OrtxStatus(kOrtxOK, "Created DeepSeek chat template.");
}

OrtxStatus TokenizerImpl::Gemma3ChatTemplate(std::string& output, bool add_generation_prompt) const {
    // Clear the output string before starting
    output.clear();

    // Handle the first system message (if any) to set the first_user_prefix
    std::string first_user_prefix = "";
    size_t start_index = 0;

    if (!messages.empty() && messages[0].at("role") == "system") {
        if (messages[0].at("content").find("{") == std::string::npos) { // String type check
            first_user_prefix = messages[0].at("content") + "\n\n";
            start_index = 1;  // Start iterating from the next message
        } else { // Handle iterable type (when content is a list)
            // Parse the message content string into JSON
            nlohmann::json message_content = nlohmann::json::parse(messages[0].at("content"));
            first_user_prefix = message_content[0]["text"].get<std::string>() + "\n\n";
            start_index = 1;  // Start iterating from the next message
        }
    }

    // Add the BOS token at the beginning of the first message
    output += bos_token;

    // Iterate through the messages starting from `start_index`
    for (size_t i = start_index; i < messages.size(); ++i) {
        const auto& message = messages[i];
        std::string role = message.at("role");
        std::string content = message.at("content");

        // Check for alternating user/assistant roles
        if ((role == "user") != (i % 2 == start_index % 2)) {
            return OrtxStatus(kOrtxErrorInvalidArgument, "Conversation roles must alternate user/assistant/user/assistant/...");
        }

        // Determine role (assistant gets converted to model)
        if (role == "assistant") {
            role = "model";
        }

        // Add start of turn with role and first user prefix if it's the first message after system message
        std::string formatted_content = "<start_of_turn>" + role + "\n";
        if (i == start_index) {
            formatted_content += first_user_prefix;
        }

        // Add content, checking if it's a string or iterable (image/text)
        if (content.find("{") == std::string::npos) { // String type content
            formatted_content += content;
        } else { // Iterable content (like image or text objects)
            // Loop through content items (assuming content is iterable)
            // Parse the message content string into JSON
            nlohmann::json message_content = nlohmann::json::parse(message.at("content"));
            for (const auto& item : message_content) {
                if (item["type"].get<std::string>() == "image") {
                    formatted_content += "<start_of_image>\n";
                } else if (item["type"].get<std::string>() == "text") {
                    formatted_content += item.at("text").get<std::string>();
                }
            }
        }

        // Close the turn
        formatted_content += "<end_of_turn>\n";
        output += formatted_content;
    }

    // Add generation prompt if necessary
    if (add_generation_prompt) {
        output += "<start_of_turn>model\n";
    }

    // Add the EOS token at the end
    output += eos_token;

    return OrtxStatus(kOrtxOK, "Created Gemma-3 chat template.");
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
        messages.push_back(message.get<std::unordered_map<std::string, std::string>>());
    }

    return messages;
}

// Constant string variables to store predefined chat template strings for popular supported models
const char PHI_VISION_CHAT_TEMPLATE[] = R"({% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt and messages[-1]['role'] != 'assistant' %}{{- '<|assistant|>\n' -}}{% endif %})";
const char PHI3_CHAT_TEMPLATE[] = R"({% for message in messages %}{% if message['role'] == 'system' %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %})";
const char PHI3_SMALL_CHAT_TEMPLATE[] = R"({{ bos_token }}{% for message in messages %}{{'<|' + message['role'] + '|>' + '\n' + message['content'] + '<|end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %})";
const char PHI3_MEDIUM_CHAT_TEMPLATE[] = R"({% for message in messages %}{% if (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif (message['role'] == 'assistant') %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %})";
const char PHI3_5_CHAT_TEMPLATE[] = R"({% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %})";
const char PHI4_CHAT_TEMPLATE[] = R"({% for message in messages %}{% if message['role'] == 'system' and 'tools' in message and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>' }}{% else %}{{ eos_token }}{% endif %})";
const char LLAMA2_CHAT_TEMPLATE[] = R"({% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %})";
const char LLAMA3_CHAT_TEMPLATE[] = R"({% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %})";
const char LLAMA3_2_CHAT_TEMPLATE[] = R"({{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now(\"%d %b %Y\") %}\n    {%- else %}\n        {%- set date_string = \"26 Jul 2024\" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n        {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n        {{- '\"parameters\": ' }}\n        {{- tool_call.arguments | tojson }}\n        {{- \"}\" }}\n        {{- \"<|eot_id|>\" }}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n)";
const char LLAMA3_3_CHAT_TEMPLATE[] = R"({{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- set date_string = \"26 Jul 2024\" %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0]['role'] == 'system' %}\n    {%- set system_message = messages[0]['content']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = \"\" %}\n{%- endif %}\n\n{#- System message + builtin tools #}\n{{- \"<|start_header_id|>system<|end_header_id|>\\n\\n\" }}\n{%- if builtin_tools is defined or tools is not none %}\n    {{- \"Environment: ipython\\n\" }}\n{%- endif %}\n{%- if builtin_tools is defined %}\n    {{- \"Tools: \" + builtin_tools | reject('equalto', 'code_interpreter') | join(\", \") + \"\\n\\n\"}}\n{%- endif %}\n{{- \"Cutting Knowledge Date: December 2023\\n\" }}\n{{- \"Today Date: \" + date_string + \"\\n\\n\" }}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- \"You have access to the following functions. To call a function, please respond with JSON for a function call.\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- \"<|eot_id|>\" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0]['content']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception(\"Cannot put tools in the first user message when there's no first user message!\") }}\n{%- endif %}\n    {{- '<|start_header_id|>user<|end_header_id|>\\n\\n' -}}\n    {{- \"Given the following functions, please respond with a JSON for a function call \" }}\n    {{- \"with its proper arguments that best answers the given prompt.\\n\\n\" }}\n    {{- 'Respond in the format {\"name\": function name, \"parameters\": dictionary of argument name and its value}.' }}\n    {{- \"Do not use variables.\\n\\n\" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- \"\\n\\n\" }}\n    {%- endfor %}\n    {{- first_user_message + \"<|eot_id|>\"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}\n        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n'+ message['content'] | trim + '<|eot_id|>' }}\n    {%- elif 'tool_calls' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception(\"This model only supports single tool-calls at once!\") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {%- if builtin_tools is defined and tool_call.name in builtin_tools %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- \"<|python_tag|>\" + tool_call.name + \".call(\" }}\n            {%- for arg_name, arg_val in tool_call.arguments | items %}\n                {{- arg_name + '=\"' + arg_val + '\"' }}\n                {%- if not loop.last %}\n                    {{- \", \" }}\n                {%- endif %}\n                {%- endfor %}\n            {{- \")\" }}\n        {%- else  %}\n            {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}\n            {{- '{\"name\": \"' + tool_call.name + '\", ' }}\n            {{- '\"parameters\": ' }}\n            {{- tool_call.arguments | tojson }}\n            {{- \"}\" }}\n        {%- endif %}\n        {%- if builtin_tools is defined %}\n            {#- This means we're in ipython mode #}\n            {{- \"<|eom_id|>\" }}\n        {%- else %}\n            {{- \"<|eot_id|>\" }}\n        {%- endif %}\n    {%- elif message.role == \"tool\" or message.role == \"ipython\" %}\n        {{- \"<|start_header_id|>ipython<|end_header_id|>\\n\\n\" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- \"<|eot_id|>\" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}\n{%- endif %}\n)";
const char DEEPSEEK_CHAT_TEMPLATE[] = R"({% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %})";
const char GEMMA3_CHAT_TEMPLATE[] = R"({{ bos_token }}\n{%- if messages[0]['role'] == 'system' -%}\n    {%- if messages[0]['content'] is string -%}\n        {%- set first_user_prefix = messages[0]['content'] + '\n\n' -%}\n    {%- else -%}\n        {%- set first_user_prefix = messages[0]['content'][0]['text'] + '\n\n' -%}\n    {%- endif -%}\n    {%- set loop_messages = messages[1:] -%}\n{%- else -%}\n    {%- set first_user_prefix = \"\" -%}\n    {%- set loop_messages = messages -%}\n{%- endif -%}\n{%- for message in loop_messages -%}\n    {%- if (message['role'] == 'user') != (loop.index0 % 2 == 0) -%}\n        {{ raise_exception(\"Conversation roles must alternate user/assistant/user/assistant/...\") }}\n    {%- endif -%}\n    {%- if (message['role'] == 'assistant') -%}\n        {%- set role = \"model\" -%}\n    {%- else -%}\n        {%- set role = message['role'] -%}\n    {%- endif -%}\n    {{ '<start_of_turn>' + role + '\n' + (first_user_prefix if loop.first else \"\") }}\n    {%- if message['content'] is string -%}\n        {{ message['content'] | trim }}\n    {%- elif message['content'] is iterable -%}\n        {%- for item in message['content'] -%}\n            {%- if item['type'] == 'image' -%}\n                {{ '<start_of_image>' }}\n            {%- elif item['type'] == 'text' -%}\n                {{ item['text'] | trim }}\n            {%- endif -%}\n        {%- endfor -%}\n    {%- else -%}\n        {{ raise_exception(\"Invalid content type\") }}\n    {%- endif -%}\n    {{ '<end_of_turn>\n' }}\n{%- endfor -%}\n{%- if add_generation_prompt -%}\n    {{'<start_of_turn>model\n'}}\n{%- endif -%}\n)";

const std::unordered_map<std::string, std::string> model_to_template_map = {
    // Phi-vision variants
    { PHI_VISION_CHAT_TEMPLATE, "microsoft/Phi-3-vision-128k-instruct" },
    { PHI_VISION_CHAT_TEMPLATE, "microsoft/Phi-3.5-vision-instruct" },

    // Phi-3 variants
    { PHI3_CHAT_TEMPLATE, "microsoft/Phi-3-mini-4k-instruct" },
    { PHI3_CHAT_TEMPLATE, "microsoft/Phi-3-mini-128k-instruct" },

    // Phi-3-5 variants
    { PHI3_5_CHAT_TEMPLATE, "microsoft/Phi-3.5-mini-instruct" },
    { PHI3_5_CHAT_TEMPLATE, "microsoft/Phi-3.5-MoE-instruct" },

    // Phi-3-small variants
    { PHI3_SMALL_CHAT_TEMPLATE, "microsoft/Phi-3-small-4k-instruct" },
    { PHI3_SMALL_CHAT_TEMPLATE, "microsoft/Phi-3-small-128k-instruct" },

    // Phi-3-medium variants
    { PHI3_MEDIUM_CHAT_TEMPLATE, "microsoft/Phi-3-medium-4k-instruct" },
    { PHI3_MEDIUM_CHAT_TEMPLATE, "microsoft/Phi-3-medium-128k-instruct" },

    // Phi-4 variants
    { PHI4_CHAT_TEMPLATE, "microsoft/Phi-4-multimodal-instruct" },

    // Llama-2 variants
    { LLAMA2_CHAT_TEMPLATE, "meta-llama/Llama-2-7b-chat-hf" },

    // Llama-3 variants
    { LLAMA3_CHAT_TEMPLATE, "meta-llama/Meta-Llama-3-8B-Instruct" },
    { LLAMA3_CHAT_TEMPLATE, "meta-llama/Meta-Llama-3-70B-Instruct" },

    // Llama-3-2 variants
    { LLAMA3_2_CHAT_TEMPLATE, "meta-llama/Llama-3.2-1B-Instruct" },
    { LLAMA3_2_CHAT_TEMPLATE, "meta-llama/Llama-3.2-3B-Instruct" },

    // Llama-3-3 variants (Llama 3.1 & 3.3 share chat template logic)
    { LLAMA3_3_CHAT_TEMPLATE, "meta-llama/Llama-3.1-8B-Instruct" },
    { LLAMA3_3_CHAT_TEMPLATE, "meta-llama/Llama-3.1-70B-Instruct" },
    { LLAMA3_3_CHAT_TEMPLATE, "meta-llama/Llama-3.3-70B-Instruct" },

    // DeepSeek variants
    { DEEPSEEK_CHAT_TEMPLATE, "deepseek-ai/DeepSeek-R1-Distill-Llama-70B" },
    { DEEPSEEK_CHAT_TEMPLATE, "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" }
};

void TokenizerImpl::InitializeChatParameters(
    const char* template_str,
    const std::vector<std::string>& custom_tools_param,
    bool tools_in_user_message_param,
    const std::string& strftime_param,
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

// ApplyChatTemplate method to choose the template logic based on chat_template
OrtxStatus TokenizerImpl::ApplyChatTemplate(
    const TokenizerImpl::MessageList& message_list,
    std::string& output,
    bool add_generation_prompt) const {

  // Find the chat_template string for this model if it is supported
  auto it = model_to_template_map.find(chat_template);
  if (it == model_to_template_map.end()) {
      // Handle other templates or custom logic here
      return OrtxStatus(kOrtxErrorNotImplemented,
        "The provided chat template is currently not supported. Custom template handling needed.");
  }

  messages = message_list;

  // Apply the corresponding chat template if it is supported
  if (chat_template == PHI4_CHAT_TEMPLATE) {
    return Phi4ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == PHI3_CHAT_TEMPLATE) {
    return Phi3ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == PHI3_5_CHAT_TEMPLATE) {
    return Phi3_5ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == PHI3_SMALL_CHAT_TEMPLATE) {
    return Phi3SmallChatTemplate(output, add_generation_prompt);
  } else if (chat_template == PHI3_MEDIUM_CHAT_TEMPLATE) {
    return Phi3MediumChatTemplate(output, add_generation_prompt);
  } else if (chat_template == PHI_VISION_CHAT_TEMPLATE) {
    return PhiVisionChatTemplate(output, add_generation_prompt);
  } else if (chat_template == LLAMA2_CHAT_TEMPLATE) {
    return Llama2ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == LLAMA3_CHAT_TEMPLATE) {
    return Llama3ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == LLAMA3_2_CHAT_TEMPLATE) {
    return Llama3_2ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == LLAMA3_3_CHAT_TEMPLATE) {
    return Llama3_3ChatTemplate(output, add_generation_prompt);
  } else if (chat_template == DEEPSEEK_CHAT_TEMPLATE) {
    return DeepSeekChatTemplate(output, add_generation_prompt);
  } else if (chat_template == GEMMA3_CHAT_TEMPLATE) {
    return Gemma3ChatTemplate(output, add_generation_prompt);
  }

  return {};
}

}  // namespace ort_extensions

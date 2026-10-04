// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cctype>
#include <string>
#include <vector>

namespace ort_extensions {
namespace detail {

inline bool TemplateUsesGemmaToolDefinitionMacro(const std::string& tmpl) {
  const auto is_identifier_start = [](char c) {
    return std::isalpha(static_cast<unsigned char>(c)) || c == '_';
  };
  const auto is_identifier_char = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
  };
  const auto is_control_marker = [](char c) {
    return c == '-' || c == '+' || c == '~';
  };
  const auto skip_space = [&tmpl](size_t& pos, size_t end) {
    while (pos < end && std::isspace(static_cast<unsigned char>(tmpl[pos]))) {
      ++pos;
    }
  };
  const auto statement_tag = [&tmpl, &skip_space, &is_control_marker](size_t begin, size_t end, const char* tag) {
    size_t pos = begin;
    if (pos < end && is_control_marker(tmpl[pos])) {
      ++pos;
    }
    skip_space(pos, end);
    const size_t tag_size = std::char_traits<char>::length(tag);
    if (end - pos < tag_size || tmpl.compare(pos, tag_size, tag) != 0) {
      return false;
    }
    pos += tag_size;
    if (pos < end && !std::isspace(static_cast<unsigned char>(tmpl[pos])) &&
        tmpl[pos] != '-' && tmpl[pos] != '+') {
      return false;
    }
    skip_space(pos, end);
    if (pos < end && is_control_marker(tmpl[pos])) {
      ++pos;
    }
    return pos == end;
  };
  const auto is_raw_end_tag = [&tmpl, &skip_space, &is_control_marker](size_t begin, size_t& after) {
    size_t pos = begin + 2;
    if (pos < tmpl.size() && is_control_marker(tmpl[pos])) {
      ++pos;
    }
    skip_space(pos, tmpl.size());
    constexpr char kEndRaw[] = "endraw";
    if (tmpl.size() - pos < sizeof(kEndRaw) - 1 ||
        tmpl.compare(pos, sizeof(kEndRaw) - 1, kEndRaw) != 0) {
      return false;
    }
    pos += sizeof(kEndRaw) - 1;
    if (pos < tmpl.size() && !std::isspace(static_cast<unsigned char>(tmpl[pos])) &&
        !is_control_marker(tmpl[pos]) && tmpl[pos] != '%') {
      return false;
    }
    skip_space(pos, tmpl.size());
    if (pos < tmpl.size() && is_control_marker(tmpl[pos])) {
      ++pos;
    }
    if (pos + 1 >= tmpl.size() || tmpl[pos] != '%' || tmpl[pos + 1] != '}') {
      return false;
    }
    after = pos + 2;
    return true;
  };

  bool in_raw = false;
  for (size_t i = 0; i < tmpl.size();) {
    if (in_raw) {
      bool found_endraw = false;
      while (i + 1 < tmpl.size()) {
        if (tmpl[i] == '{' && tmpl[i + 1] == '%') {
          size_t after = 0;
          if (is_raw_end_tag(i, after)) {
            i = after;
            found_endraw = true;
            break;
          }
        }
        ++i;
      }
      if (!found_endraw) {
        return false;
      }
      if (found_endraw) {
        in_raw = false;
      }
      continue;
    }

    if (i + 1 < tmpl.size() && tmpl[i] == '{' && tmpl[i + 1] == '#') {
      const auto comment_end = tmpl.find("#}", i + 2);
      if (comment_end == std::string::npos) {
        return false;
      }
      i = comment_end + 2;
      continue;
    }
    if (i + 1 >= tmpl.size() || tmpl[i] != '{' || (tmpl[i + 1] != '{' && tmpl[i + 1] != '%')) {
      ++i;
      continue;
    }

    const bool is_statement = tmpl[i + 1] == '%';
    const size_t block_begin = i + 2;
    size_t block_end = block_begin;
    size_t brace_depth = 0;
    char quote = '\0';
    while (block_end + 1 < tmpl.size()) {
      const char c = tmpl[block_end];
      if (quote != '\0') {
        if (c == '\\' && block_end + 1 < tmpl.size()) {
          block_end += 2;
          continue;
        }
        if (c == quote) {
          quote = '\0';
        }
      } else if (c == '\'' || c == '"') {
        quote = c;
      } else if (!is_statement && c == '{') {
        ++brace_depth;
      } else if (!is_statement && c == '}') {
        if (brace_depth > 0) {
          --brace_depth;
        } else if (tmpl[block_end + 1] == '}') {
          break;
        }
      } else if (is_statement && c == '%' && tmpl[block_end + 1] == '}') {
        break;
      }
      ++block_end;
    }
    if (block_end + 1 >= tmpl.size()) {
      return false;
    }

    if (is_statement) {
      size_t cursor = block_begin;
      if (cursor < block_end && is_control_marker(tmpl[cursor])) {
        ++cursor;
      }
      skip_space(cursor, block_end);

      if (statement_tag(block_begin, block_end, "raw")) {
        in_raw = true;
        i = block_end + 2;
        continue;
      }

      constexpr char kMacro[] = "macro";
      if (block_end - cursor >= sizeof(kMacro) - 1 &&
          tmpl.compare(cursor, sizeof(kMacro) - 1, kMacro) == 0 &&
          (cursor + sizeof(kMacro) - 1 == block_end ||
           !is_identifier_char(tmpl[cursor + sizeof(kMacro) - 1]))) {
        cursor += sizeof(kMacro) - 1;
        skip_space(cursor, block_end);

        constexpr char kGemmaMacro[] = "format_function_declaration";
        if (block_end - cursor >= sizeof(kGemmaMacro) - 1 &&
            tmpl.compare(cursor, sizeof(kGemmaMacro) - 1, kGemmaMacro) == 0 &&
            (cursor + sizeof(kGemmaMacro) - 1 == block_end ||
             !is_identifier_char(tmpl[cursor + sizeof(kGemmaMacro) - 1]))) {
          cursor += sizeof(kGemmaMacro) - 1;
          skip_space(cursor, block_end);
          if (cursor < block_end && tmpl[cursor++] == '(') {
            bool found_tool_data = false;
            bool valid_signature = true;
            bool need_parameter = true;
            std::vector<char> delimiters;

            while (cursor < block_end) {
              skip_space(cursor, block_end);
              if (cursor < block_end && tmpl[cursor] == ')' && delimiters.empty()) {
                ++cursor;
                need_parameter = false;
                break;
              }
              if (!need_parameter) {
                valid_signature = false;
                break;
              }
              if (cursor >= block_end || !is_identifier_start(tmpl[cursor])) {
                valid_signature = false;
                break;
              }

              const size_t name_begin = cursor++;
              while (cursor < block_end && is_identifier_char(tmpl[cursor])) {
                ++cursor;
              }
              found_tool_data |= tmpl.compare(name_begin, cursor - name_begin, "tool_data") == 0;
              skip_space(cursor, block_end);

              if (cursor < block_end && tmpl[cursor] == '=') {
                ++cursor;
                char default_quote = '\0';
                bool has_default_value = false;
                while (cursor < block_end) {
                  const char c = tmpl[cursor];
                  if (default_quote != '\0') {
                    has_default_value = true;
                    if (c == '\\' && cursor + 1 < block_end) {
                      cursor += 2;
                      continue;
                    }
                    if (c == default_quote) {
                      default_quote = '\0';
                    }
                    ++cursor;
                    continue;
                  }
                  if (c == '\'' || c == '"') {
                    default_quote = c;
                    has_default_value = true;
                    ++cursor;
                    continue;
                  }
                  if (c == '(' || c == '[' || c == '{') {
                    delimiters.push_back(c);
                    has_default_value = true;
                    ++cursor;
                    continue;
                  }
                  if (c == ')' || c == ']' || c == '}') {
                    if (delimiters.empty()) {
                      if (c == ')') {
                        break;
                      }
                      valid_signature = false;
                      break;
                    }
                    const char expected_open = c == ')' ? '(' : (c == ']' ? '[' : '{');
                    if (delimiters.back() != expected_open) {
                      valid_signature = false;
                      break;
                    }
                    delimiters.pop_back();
                    has_default_value = true;
                    ++cursor;
                    continue;
                  }
                  if (c == ',' && delimiters.empty()) {
                    break;
                  }
                  has_default_value |= !std::isspace(static_cast<unsigned char>(c));
                  ++cursor;
                }
                if (!valid_signature || default_quote != '\0' || !delimiters.empty() || !has_default_value) {
                  valid_signature = false;
                  break;
                }
              }

              skip_space(cursor, block_end);
              if (cursor < block_end && tmpl[cursor] == ',') {
                ++cursor;
                need_parameter = true;
              } else if (cursor < block_end && tmpl[cursor] == ')') {
                ++cursor;
                need_parameter = false;
                break;
              } else {
                valid_signature = false;
                break;
              }
            }

            skip_space(cursor, block_end);
            if (cursor < block_end && is_control_marker(tmpl[cursor])) {
              ++cursor;
            }
            if (valid_signature && !need_parameter && cursor == block_end && found_tool_data) {
              return true;
            }
          }
        }
      }
    }
    i = block_end + 2;
  }
  return false;
}

}  // namespace detail
}  // namespace ort_extensions

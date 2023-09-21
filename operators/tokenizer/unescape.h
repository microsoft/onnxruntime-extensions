// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "ustring.h"
#include <vector>

namespace ort_extensions {

inline bool IsDigit(char c) { return c >= '0' && c <= '9'; }
inline bool IsHexDigit(char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

inline unsigned int hex_digit_to_int(char c) {
  unsigned int x = static_cast<unsigned char>(c);
  if (x > '9') {
    x += 9;
  }
  return x & 0xf;
}

inline bool IsSurrogate(char32_t c) {
  return c >= 0xD800 && c <= 0xDFFF;
}

// Unescape a Python escaped string
inline bool Unescape(const std::string_view& source, std::string& unescaped, bool is_bytes) {

  // reserve enough space for the worst case, and final size will be calculated at the end.
  unescaped.resize(source.length());
  char* d = unescaped.data();
  const char* p = source.data();
  const char* end = p + source.size();
  const char* last_byte = end - 1;

  while (p == d && p < end && *p != '\\') p++, d++;

  while (p < end) {
    if (*p != '\\') {
      *d++ = *p++;
    } else {
      if (++p > last_byte) {
        return false;
      }
      switch (*p) {
        case 'n':
          *d++ = '\n';
          break;
        case 'r':
          *d++ = '\r';
          break;
        case 't':
          *d++ = '\t';
          break;
          break;
        case '\\':
          *d++ = '\\';
          break;
        case '\'':
          *d++ = '\'';
          break;
        case '"':
          *d++ = '\"';
          break;
        case 'x':
        case 'X': {
          if (p >= last_byte) {
            return false;
          } else if (!IsHexDigit(static_cast<unsigned char>(p[1]))) {
            return false;
          }
          unsigned int ch = 0;
          const char* hex_start = p;
          while (p < last_byte &&
                 IsHexDigit(static_cast<unsigned char>(p[1])))
            ch = (ch << 4) + hex_digit_to_int(*++p);
          if (ch > 0xFF && !is_bytes) {
            return false;
          }
          if (is_bytes) {
            *d++ = static_cast<char>(ch);
          } else {
            d += ustring::EncodeUTF8Char(d, static_cast<char32_t>(ch));
          }
          break;
        }
        case 'u': {
          char32_t rune = 0;
          const char* hex_start = p;
          if (p + 4 >= end) {
            return false;
          }
          for (int i = 0; i < 4; ++i) {
            if (IsHexDigit(static_cast<unsigned char>(p[1]))) {
              rune = (rune << 4) + hex_digit_to_int(*++p);
            } else {
              return false;
            }
          }
          if (IsSurrogate(rune)) {
            return false;
          }
          d += ustring::EncodeUTF8Char(d, rune);
          break;
        }
        case 'U': {
          char32_t rune = 0;
          const char* hex_start = p;
          if (p + 8 >= end) {
            return false;
          }
          for (int i = 0; i < 8; ++i) {
            if (IsHexDigit(static_cast<unsigned char>(p[1]))) {
              uint32_t newrune = (rune << 4) + hex_digit_to_int(*++p);
              if (newrune > 0x10FFFF) {
                return false;
              } else {
                rune = newrune;
              }
            } else {
              return false;
            }
          }
          if (IsSurrogate(rune)) {
            return false;
          }
          d += ustring::EncodeUTF8Char(d, rune);
          break;
        }
        default: {
          return false;
        }
      }
      p++;
    }
  }

  unescaped.resize(d - unescaped.data());
  return true;
}

inline bool UnquoteString(const std::string& str, std::string& unquoted) {
  bool is_bytes = false;
  int idx_0 = 0;
  if (str.front() == 'b') {
    is_bytes = true;
    idx_0 = 1;
  }
  std::string str_view(str.data() + idx_0, str.length() - idx_0);
  if (str_view.length() < 2) {
    return false;
  }

  if ((str_view.front() != '\"' && str_view.front() != '\'') || str_view.back() != str.back()) {
    return false;
  }

  // unescape the string
  return Unescape(std::string_view(str_view.data() + 1, str_view.length() - 2), unquoted, is_bytes);
}

}  // namespace ort_extensions

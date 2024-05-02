// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <vector>
#include <string_view>

// ustring needs a new implementation, due to the std::codecvt deprecation.
// Wrap u32string with ustring, in case we will use other implementation in the future
class ustring : public std::u32string {
 public:
  ustring() = default;

  explicit ustring(const char* str) { assign(FromUTF8(str)); }

  explicit ustring(const std::string& str) { assign(FromUTF8(str)); }

  explicit ustring(const std::string_view& str) { assign(FromUTF8(str)); }

  explicit ustring(const char32_t* str) : std::u32string(str) {}

  explicit ustring(const std::u32string_view& str) : std::u32string(str) {}

  explicit operator std::string() const { return ToUTF8(*this); }

  static size_t EncodeUTF8Char(char* buffer, char32_t utf8_char) {
    if (utf8_char <= 0x7F) {
      *buffer = static_cast<char>(utf8_char);
      return 1;
    } else if (utf8_char <= 0x7FF) {
      buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[0] = static_cast<char>(0xC0 | utf8_char);
      return 2;
    } else if (utf8_char <= 0xFFFF) {
      buffer[2] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[0] = static_cast<char>(0xE0 | utf8_char);
      return 3;
    } else {
      buffer[3] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[2] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[1] = static_cast<char>(0x80 | (utf8_char & 0x3F));
      utf8_char >>= 6;
      buffer[0] = static_cast<char>(0xF0 | utf8_char);
      return 4;
    }
  }

  static std::string EncodeUTF8Char(char32_t utf8_char) {
    char utf8_buf[5];  // one extra space for zero
    auto clen = EncodeUTF8Char(utf8_buf, utf8_char);
    utf8_buf[clen] = 0;
    return std::string(utf8_buf);
  }

  static size_t UTF8Len(char byte1) {
    const size_t lookup[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4};
    uint8_t highbits = static_cast<uint8_t>(byte1) >> 4;
    return lookup[highbits];
  }

  static size_t UTF8Len(char32_t codepoint) {
    if (codepoint <= 0x7F) {
      return 1;
    } else if (codepoint <= 0x7FF) {
      return 2;
    } else if (codepoint <= 0xFFFF) {
      return 3;
    } else {
      return 4;
    }
  }

  static bool ValidateUTF8(const std::string& data) {
    const unsigned char* s = reinterpret_cast<const unsigned char*>(data.c_str());
    const unsigned char* s_end = s + data.size();
    if (*s_end != '\0')
      return false;

    while (*s) {
      if (*s < 0x80)
        /* 0xxxxxxx */
        s++;
      else if ((s[0] & 0xe0) == 0xc0) {
        /* 110XXXXx 10xxxxxx */
        if (s + 1 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[0] & 0xfe) == 0xc0) /* overlong? */
          return false;
        else
          s += 2;
      } else if ((s[0] & 0xf0) == 0xe0) {
        /* 1110XXXX 10Xxxxxx 10xxxxxx */
        if (s + 2 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[2] & 0xc0) != 0x80 ||
            (s[0] == 0xe0 && (s[1] & 0xe0) == 0x80) || /* overlong? */
            (s[0] == 0xed && (s[1] & 0xe0) == 0xa0) || /* surrogate? */
            (s[0] == 0xef && s[1] == 0xbf &&
             (s[2] & 0xfe) == 0xbe)) /* U+FFFE or U+FFFF? */
          return false;
        else
          s += 3;
      } else if ((s[0] & 0xf8) == 0xf0) {
        /* 11110XXX 10XXxxxx 10xxxxxx 10xxxxxx */
        if (s + 3 >= s_end) {
          return false;
        }
        if ((s[1] & 0xc0) != 0x80 ||
            (s[2] & 0xc0) != 0x80 ||
            (s[3] & 0xc0) != 0x80 ||
            (s[0] == 0xf0 && (s[1] & 0xf0) == 0x80) ||    /* overlong? */
            (s[0] == 0xf4 && s[1] > 0x8f) || s[0] > 0xf4) /* > U+10FFFF? */
          return false;
        else
          s += 4;
      } else
        return false;
    }

    return true;
  }

 private:
  using u32string = std::u32string;
  static u32string FromUTF8(const std::string_view& utf8) {
    u32string ucs32;
    ucs32.reserve(utf8.length() / 2);  // a rough estimation for less memory allocation.
    for (size_t i = 0; i < utf8.size();) {
      char32_t codepoint = 0;
      if ((utf8[i] & 0x80) == 0) {
        codepoint = utf8[i];
        i++;
      } else if ((utf8[i] & 0xE0) == 0xC0) {
        codepoint = ((utf8[i] & 0x1F) << 6) | (utf8[i + 1] & 0x3F);
        i += 2;
      } else if ((utf8[i] & 0xF0) == 0xE0) {
        codepoint = ((utf8[i] & 0x0F) << 12) | ((utf8[i + 1] & 0x3F) << 6) | (utf8[i + 2] & 0x3F);
        i += 3;
      } else {
        codepoint = ((utf8[i] & 0x07) << 18) | ((utf8[i + 1] & 0x3F) << 12) | ((utf8[i + 2] & 0x3F) << 6) | (utf8[i + 3] & 0x3F);
        i += 4;
      }
      ucs32.push_back(codepoint);
    }
    return ucs32;
  }

  static std::string ToUTF8(const u32string& ucs32) {
    std::string utf8;
    utf8.reserve(ucs32.length() * 4);
    for (char32_t codepoint : ucs32) {
      utf8 += EncodeUTF8Char(codepoint);
    }

    return utf8;
  }
};

namespace std {
template <>
struct hash<ustring> {
  size_t operator()(const ustring& __str) const noexcept {
    hash<u32string> standard_hash;
    return standard_hash(static_cast<u32string>(__str));
  }
};
}  // namespace std

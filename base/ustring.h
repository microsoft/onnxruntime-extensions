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

  explicit operator std::u32string() const { return *this; }

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

  static bool ValidateUTF8(const std::string& data) {
    int cnt = 0;
    for (auto i = 0; i < data.size(); i++) {
      int x = data[i];
      if (!cnt) {
        if ((x >> 5) == 0b110) {
          cnt = 1;
        } else if ((x >> 4) == 0b1110) {
          cnt = 2;
        } else if ((x >> 3) == 0b11110) {
          cnt = 3;
        } else if ((x >> 7) != 0) {
          return false;
        }
      } else {
        if ((x >> 6) != 0b10) return false;
        cnt--;
      }
    }
    return cnt == 0;
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

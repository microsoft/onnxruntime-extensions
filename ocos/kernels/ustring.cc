// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>
#include "ustring.hpp"

ustring::ustring(): std::u32string() {
}

ustring::ustring(char* str) {
  utf8_converter str_cvt;
  assign(str_cvt.from_bytes(str));
}

ustring::ustring(const char* str) {
  utf8_converter str_cvt;
  assign(str_cvt.from_bytes(str));
}

ustring::ustring(std::string& str) {
  utf8_converter str_cvt;
  assign(str_cvt.from_bytes(str));
}

ustring::ustring(const std::string& str) {
  utf8_converter str_cvt;
  assign(str_cvt.from_bytes(str));
}

ustring::ustring(char32_t* str):std::u32string(str) {}

ustring::ustring(const char32_t* str):std::u32string(str) {}

ustring::ustring(std::u32string& str):std::u32string(str) {}

ustring::ustring(std::u32string&& str):std::u32string(str) {}

ustring::ustring(const std::u32string& str):std::u32string(str) {}

ustring::ustring(const std::u32string&& str):std::u32string(str) {}

ustring::ustring(std::u32string_view& str):std::u32string(str) {}

ustring::ustring(std::u32string_view&& str):std::u32string(str) {}

ustring::ustring(const std::u32string_view& str):std::u32string(str) {}

ustring::ustring(const std::u32string_view&& str):std::u32string(str) {}

ustring::operator std::string() {
  utf8_converter str_cvt;
  return str_cvt.to_bytes(*this);
}

ustring::operator std::string() const {
  utf8_converter str_cvt;
  return str_cvt.to_bytes(*this);
}

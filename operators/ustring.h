// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <string>
#include <locale>
#include <functional>

#define _SILENCE_CXX17_CODECVT_HEADER_DEPRECATION_WARNING 1
#include <codecvt>

#include "ocos.h"

// Wrap u32string with ustring, in case we will use other implementation in the future
class ustring : public std::u32string {
 public:
  ustring();
  explicit ustring(char* str);
  explicit ustring(const char* str);
  explicit ustring(std::string& str);
  explicit ustring(const std::string& str);
  explicit ustring(char32_t* str);
  explicit ustring(const char32_t* str);
  explicit ustring(std::u32string& str);
  explicit ustring(std::u32string&& str);
  explicit ustring(const std::u32string& str);
  explicit ustring(const std::u32string&& str);
  explicit ustring(std::string_view& str);
  explicit ustring(const std::string_view& str);
  explicit ustring(std::u32string_view& str);
  explicit ustring(std::u32string_view&& str);
  explicit ustring(const std::u32string_view& str);
  explicit ustring(const std::u32string_view&& str);

  explicit operator std::string();
  explicit operator std::string() const;

 private:
  using utf8_converter = std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t>;
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


void GetTensorMutableDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                                const OrtValue* value, std::vector<ustring>& output);

void FillTensorDataString(const OrtApi& api, OrtW::CustomOpApi& ort, OrtKernelContext* context,
                          const std::vector<ustring>& value, OrtValue* output);

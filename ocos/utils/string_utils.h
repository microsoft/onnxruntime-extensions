// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <iostream>
#include <sstream>
#include <vector>

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept {
  ss << "[";
  for (int i = 0; i < t.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << t[i];
  }
  ss << "]";
}

template <typename T, typename... Args>
void MakeStringInternal(std::ostringstream& ss, const T& t, const Args&... args) noexcept {
  MakeStringInternal(ss, t);
  MakeStringInternal(ss, args...);
}

template <typename... Args>
std::string MakeString(const Args&... args) {
  std::ostringstream ss;
  MakeStringInternal(ss, args...);
  return std::string(ss.str());
}

std::vector<std::string_view> SplitString(const std::string_view& str, const std::string_view& seps, bool remove_empty_entries = false);

void char2unicode(const std::string& src, std::vector<uint32_t>& result);
void unicode2char(const std::vector<uint32_t>& src, std::string& result);

void to_bytes(const std::vector<std::wstring>& src, std::vector<std::string>& dest);
void from_bytes(const std::vector<std::string>& src, std::vector<std::wstring>& dest);

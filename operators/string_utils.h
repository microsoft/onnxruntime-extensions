// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <sstream>
#include <vector>
#include "ocos.h"

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<int64_t>& t) noexcept {
  ss << "[";
  for (size_t i = 0; i < t.size(); i++) {
    if (i != 0) {
      ss << ", ";
    }
    ss << t[i];
  }
  ss << "]";
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const OrtTensorDimensions& t) noexcept {
  MakeStringInternal(ss, static_cast<const std::vector<int64_t>&>(t));
}

template <>
inline void MakeStringInternal(std::ostringstream& ss, const std::vector<std::string>& t) noexcept {
  ss << "[";
  for (size_t i = 0; i < t.size(); i++) {
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

bool IsCJK(char32_t c);

bool IsAccent(char32_t c);

bool IsSpace(char32_t c);

bool IsPunct(char32_t c);

bool IsControl(char32_t c);

char32_t ToLower(char32_t c);

char32_t StripAccent(char32_t c);

uint64_t Hash64(const char* data, size_t n, uint64_t seed);

inline uint64_t Hash64(const char* data, size_t n) {
  return Hash64(data, n, 0xDECAFCAFFE);
}

uint64_t Hash64Fast(const char* data, size_t n);

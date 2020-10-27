// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <iostream>
#include <sstream>

template <typename T>
inline void MakeStringInternal(std::ostringstream& ss, const T& t) noexcept {
  ss << t;
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <locale>
#include <optional>
#include <string>
#include <sstream>
#include "string_utils.h"
#ifdef _WIN32
#include <Windows.h>
#endif

#define ORTX_RETURN_IF_ERROR(expr) \
  do {                             \
    auto _status = (expr);         \
    if (_status != nullptr) {      \
      return _status;              \
    }                              \
  } while (0)

template <typename T>
bool TryParseStringWithClassicLocale(std::string_view str, T& value) {
  if constexpr (std::is_integral<T>::value && std::is_unsigned<T>::value) {
    // if T is unsigned integral type, reject negative values which will wrap
    if (!str.empty() && str[0] == '-') {
      return false;
    }
  }

  // don't allow leading whitespace
  if (!str.empty() && std::isspace(str[0], std::locale::classic())) {
    return false;
  }

  std::istringstream is{std::string{str}};
  is.imbue(std::locale::classic());
  T parsed_value{};

  const bool parse_successful =
      is >> parsed_value &&
      is.get() == std::istringstream::traits_type::eof();  // don't allow trailing characters
  if (!parse_successful) {
    return false;
  }

  value = std::move(parsed_value);
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, std::string& value) {
  value = str;
  return true;
}

inline bool TryParseStringWithClassicLocale(std::string_view str, bool& value) {
  if (str == "0" || str == "False" || str == "false") {
    value = false;
    return true;
  }

  if (str == "1" || str == "True" || str == "true") {
    value = true;
    return true;
  }

  return false;
}

template <typename T>
std::optional<T> ParseEnvironmentVariable(const std::string& name) {
  std::string buffer;
#ifdef _WIN32
  constexpr size_t kBufferSize = 32767;

  // Create buffer to hold the result
  buffer.resize(kBufferSize, '\0');

  // The last argument is the size of the buffer pointed to by the lpBuffer parameter, including the null-terminating character, in characters.
  // If the function succeeds, the return value is the number of characters stored in the buffer pointed to by lpBuffer, not including the terminating null character.
  // Therefore, If the function succeeds, kBufferSize should be larger than char_count.
  auto char_count = GetEnvironmentVariableA(name.c_str(), buffer.data(), kBufferSize);

  if (kBufferSize > char_count) {
    buffer.resize(char_count);
  } else {
    // Else either the call was failed, or the buffer wasn't large enough.
    // TODO: Understand the reason for failure by calling GetLastError().
    // If it is due to the specified environment variable being found in the environment block,
    // GetLastError() returns ERROR_ENVVAR_NOT_FOUND.
    // For now, we assume that the environment variable is not found.
    buffer.clear();
  }
#else
  char* val = getenv(name.c_str());
  buffer = (val == nullptr) ? std::string() : std::string(val);
#endif
  T parsed_value;
  if (!TryParseStringWithClassicLocale(buffer, parsed_value)) {
    OrtW::Exception(MakeString("Failed to parse environment variable - name: ", name, ", value: ", buffer), OrtErrorCode::ORT_FAIL);
  }
  return parsed_value;
}

template <typename T>
T ParseEnvironmentVariableWithDefault(const std::string& name, const T& default_value) {
  const auto parsed = ParseEnvironmentVariable<T>(name);
  if (parsed.has_value()) {
    return *parsed;
  }

  return default_value;
}

inline bool IsScalarOr1ElementVector(size_t num_dimensions, int64_t shape_size) {
  if (num_dimensions == 0 || (num_dimensions == 1 && shape_size == 1)) return true;
  return false;
}

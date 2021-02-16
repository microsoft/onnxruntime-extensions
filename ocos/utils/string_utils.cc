// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef _MSC_VER
#include <codecvt>
#include <locale.h>
#elif defined(__APPLE__) or defined(__ANDROID__)
#include <codecvt>
#else
#include <limits>
#include <iconv.h>
#endif  // _MSC_VER

#include <locale>
#include <functional>
#include <unordered_set>
#include <algorithm>
#include "string_utils.h"

// constants

#define ORT_DISALLOW_COPY(TypeName) TypeName(const TypeName&) = delete

#define ORT_DISALLOW_ASSIGNMENT(TypeName) TypeName& operator=(const TypeName&) = delete

#define ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName) \
  ORT_DISALLOW_COPY(TypeName);                     \
  ORT_DISALLOW_ASSIGNMENT(TypeName)

#define ORT_DISALLOW_MOVE(TypeName) \
  TypeName(TypeName&&) = delete;    \
  TypeName& operator=(TypeName&&) = delete

#define ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TypeName) \
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TypeName);           \
  ORT_DISALLOW_MOVE(TypeName)

std::vector<std::string_view> SplitString(const std::string_view& str, const std::string_view& seps, bool remove_empty_entries) {
  std::vector<std::string_view> result;
  std::string ::size_type pre_pos = 0;

  while (true) {
    auto next_pos = str.find_first_of(seps, pre_pos);

    if (next_pos == std::string::npos) {
      auto sub_str = str.substr(pre_pos, next_pos);
      // sub_str is empty means the last sep reach the end of string
      if (!sub_str.empty()) {
        result.push_back(sub_str);
      }

      break;
    }

    if (pre_pos != next_pos || !remove_empty_entries) {
      auto sub_str = str.substr(pre_pos, next_pos - pre_pos);
      result.push_back(sub_str);
    }

    pre_pos = next_pos + 1;
  }

  return result;
}

/////////
// Locale
/////////

// We need to specialize for MS as there is
// a std::locale creation bug that affects different
// environments in a different way
#ifdef _MSC_VER

class Locale {
 public:
  explicit Locale(const std::string& name)
      : loc_(nullptr) {
    loc_ = _create_locale(LC_CTYPE, name.c_str());
    if (loc_ == nullptr) {
      throw std::runtime_error(MakeString(
          "Failed to construct locale with name:",
          name, ":", ":Please, install necessary language-pack-XX and configure locales"));
    }
  }

  ~Locale() {
    if (loc_ != nullptr) {
      _free_locale(loc_);
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Locale);

 private:
  _locale_t loc_;
};

const std::string default_locale("en-US");

#else  // MS_VER

class Locale {
 public:
  explicit Locale(const std::string& name) {
    try {
      loc_ = std::locale(name.c_str());
    } catch (const std::runtime_error& e) {
      throw std::runtime_error(MakeString(
          "Failed to construct locale with name:",
          name, ":", e.what(), ":Please, install necessary language-pack-XX and configure locales"));
    }
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Locale);

 private:
  std::locale loc_;
};

const std::string default_locale("en_US.UTF-8");  // All non-MS

#endif  // MS_VER

////////////
// converter
////////////

#ifdef _MSC_VER
using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;
#else
const std::string conv_error("Conversion Error");
const std::wstring wconv_error(L"Conversion Error");

#if defined(__APPLE__) or defined(__ANDROID__)
using Utf8Converter = std::wstring_convert<std::codecvt_utf8<wchar_t>>;
#else

// All others (Linux)
class Utf8Converter {
 public:
  Utf8Converter(const std::string&, const std::wstring&) {}

  std::wstring from_bytes(const std::string& s) const {
    std::wstring result;
    if (s.empty()) {
      return result;
    }
    // Order of arguments is to, from
    auto icvt = iconv_open("WCHAR_T", "UTF-8");
    // CentOS is not happy with -1
    if (std::numeric_limits<iconv_t>::max() == icvt) {
      return wconv_error;
    }

    char* iconv_in = const_cast<char*>(s.c_str());
    size_t iconv_in_bytes = s.length();
    // Temporary buffer assumes 1 byte to 1 wchar_t
    // to make sure it is enough.
    const size_t buffer_len = iconv_in_bytes * sizeof(wchar_t);
    auto buffer = std::make_unique<char[]>(buffer_len);
    char* iconv_out = buffer.get();
    size_t iconv_out_bytes = buffer_len;
    auto ret = iconv(icvt, &iconv_in, &iconv_in_bytes, &iconv_out, &iconv_out_bytes);
    if (static_cast<size_t>(-1) == ret) {
      result = wconv_error;
    } else {
      size_t converted_bytes = buffer_len - iconv_out_bytes;
      assert((converted_bytes % sizeof(wchar_t)) == 0);
      result.assign(reinterpret_cast<const wchar_t*>(buffer.get()), converted_bytes / sizeof(wchar_t));
    }
    iconv_close(icvt);
    return result;
  }

  std::string to_bytes(const std::wstring& wstr) const {
    std::string result;
    if (wstr.empty()) {
      return result;
    }
    // Order of arguments is to, from
    auto icvt = iconv_open("UTF-8", "WCHAR_T");
    // CentOS is not happy with -1
    if (std::numeric_limits<iconv_t>::max() == icvt) {
      return conv_error;
    }

    // I hope this does not modify the incoming buffer
    wchar_t* non_const_in = const_cast<wchar_t*>(wstr.c_str());
    char* iconv_in = reinterpret_cast<char*>(non_const_in);
    size_t iconv_in_bytes = wstr.length() * sizeof(wchar_t);
    // Temp buffer, assume every code point converts into 3 bytes, this should be enough
    // We do not convert terminating zeros
    const size_t buffer_len = wstr.length() * 3;
    auto buffer = std::make_unique<char[]>(buffer_len);

    char* iconv_out = buffer.get();
    size_t iconv_out_bytes = buffer_len;
    auto ret = iconv(icvt, &iconv_in, &iconv_in_bytes, &iconv_out, &iconv_out_bytes);
    if (static_cast<size_t>(-1) == ret) {
      result = conv_error;
    } else {
      size_t converted_len = buffer_len - iconv_out_bytes;
      result.assign(buffer.get(), converted_len);
    }
    iconv_close(icvt);
    return result;
  }
};

#endif  // __APPLE__
#endif

void to_bytes(const std::vector<std::wstring>& src, std::vector<std::string>& dest) {
  Utf8Converter converter;
  dest.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dest[i] = converter.to_bytes(src[i]);
}

void from_bytes(const std::vector<std::string>& src, std::vector<std::wstring>& dest) {
  Utf8Converter converter;
  dest.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dest[i] = converter.from_bytes(src[i]);
}

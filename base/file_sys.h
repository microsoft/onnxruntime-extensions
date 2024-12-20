// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // Exclude rarely-used stuff from Windows headers
#endif

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <string>
#include <fstream>


namespace ort_extensions {

class path {
 public:
  path() = default;
  explicit path(const std::string& path) : path_(path) {
#ifdef _WIN32
    w_path_ = to_wstring();
#endif  // _WIN32
  };

#ifdef _WIN32
  explicit path(const std::wstring& wpath) {
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string utf8_str(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, &utf8_str[0], size_needed, nullptr, nullptr);
    path_ = utf8_str;
  }
#endif  // _WIN32

  static constexpr char separator =
#ifdef _WIN32
      '\\';
#else
      '/';
#endif

  using ios_base = std::ios_base;
  std::ifstream open(ios_base::openmode mode = ios_base::in) const {
    // if Windows, need to convert the string to UTF-16
#ifdef _WIN32
    return std::ifstream(w_path_, mode);
#else
    return std::ifstream(path_, mode);
#endif  // _WIN32
  }

  const std::string& string() const {
    return path_;
  }

  path join(const std::string& str) const {
    return path(path_ + separator + str);
  }

  path operator/(const std::string& str) const {
    return join(str);
  }

  path operator/(const path& path) {
    return join(path.path_);
  }

  bool is_regular_file() const {
    auto info = get_stat();
    return (info.st_mode & S_IFREG) != 0;
  }

  bool is_directory() const {
    auto info = get_stat();
    return (info.st_mode & S_IFDIR) != 0;
  }

  std::string extension() const {
    return path_.substr(path_.find_last_of('.'));
  }

  std::string parent_path() const {
    std::string sep = {separator};
#ifdef _WIN32
    sep += "/";
#endif  // _WIN32
    auto pos = path_.find_last_of(sep);
    if (pos == std::string::npos) {
      return "";
    }
    return path_.substr(0, pos);
  }

#ifdef _WIN32
  struct _stat64 get_stat() const {
    struct _stat64 info;
    if (_wstat64(w_path_.c_str(), &info) != 0) {
      return {};
    }
    return info;
  }
#else
  struct stat get_stat() const {
    struct stat info;
    if (stat(path_.c_str(), &info) != 0) {
      return {};
    }
    return info;
  }
#endif  // _WIN32

  bool exists() const {
    auto info = get_stat();
    return (info.st_mode & S_IFMT) != 0;
  }

 private:
  std::string path_;
#ifdef _WIN32
  std::wstring w_path_;

  std::wstring to_wstring() const {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, nullptr, 0);
    std::wstring utf16_str(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, &utf16_str[0], size_needed);
    return utf16_str;
  }
#endif  // _WIN32
};

}  // namespace ort_extensions

namespace ortx = ort_extensions;

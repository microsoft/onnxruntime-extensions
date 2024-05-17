// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <string>
#include <fstream>

namespace ort_extensions {

class path {
 public:
  path() = default;
  path(const std::string& path) : path_(path){};

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
    return std::ifstream(to_wstring(), mode);
#else
    return std::ifstream(path_, mode);
#endif  // _WIN32
  }

  const std::string& string() const {
    return path_;
  }

  path join(const std::string& path) const {
    return path_ + separator + path;
  }

  path operator/(const std::string& path) const {
    return join(path);
  }

  path operator/(const path& path) {
    return join(path.path_);
  }

  bool is_directory() const {
#ifdef _WIN32
    struct _stat64 info;
    if (_wstat64(to_wstring().c_str(), &info) != 0) {
      return false;
    }
#else
    struct stat info;
    if (stat(path_.c_str(), &info) != 0) {
      return false;
    }
#endif  // _WIN32
    return (info.st_mode & S_IFDIR) != 0;
  }

 private:
  std::string path_;

#ifdef _WIN32
  std::wstring to_wstring() const {
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, nullptr, 0);
    std::wstring utf16_str(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, path_.c_str(), -1, &utf16_str[0], size_needed);
    return utf16_str;
  }
#endif  // _WIN32
};

}  // namespace ort_extensions

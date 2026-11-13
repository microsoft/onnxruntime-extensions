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
#include <cwchar>
#endif  // _WIN32

#include <sys/stat.h>

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>

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
  explicit path(const std::wstring& wpath) : path_(from_wstring(wpath)), w_path_(wpath) {}
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

  const std::string& string() const { return path_; }

  path join(const std::string& str) const {
    if (path_.empty()) {
      return path(str);
    }
    return path(path_ + separator + str);
  }

  path operator/(const std::string& str) const { return join(str); }

  path operator/(const path& rhs) const { return join(rhs.path_); }

  bool is_regular_file() const {
    auto info = get_stat();
    return (info.st_mode & S_IFREG) != 0;
  }

  bool is_directory() const {
    auto info = get_stat();
    return (info.st_mode & S_IFDIR) != 0;
  }

  std::string extension() const {
    auto sep_pos = path_.find_last_of("/\\");
    auto dot_pos = path_.find_last_of('.');
    if (dot_pos == std::string::npos || (sep_pos != std::string::npos && dot_pos < sep_pos)) {
      return "";
    }
    return path_.substr(dot_pos);
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

  // Validates lexically that rel_path is a relative file path without parent traversal,
  // drive/root prefixes, embedded NULs, or Windows device/stream syntax.
  static bool is_safe_relative_path(const std::string& rel_path) {
    if (rel_path.empty() || rel_path.find('\0') != std::string::npos) {
      return false;
    }
    if (rel_path.front() == '/' || rel_path.front() == '\\' || rel_path.back() == '/' || rel_path.back() == '\\') {
      return false;
    }
    if (rel_path.size() >= 2 && std::isalpha(static_cast<unsigned char>(rel_path[0])) && rel_path[1] == ':') {
      return false;
    }
#ifdef _WIN32
    if (rel_path.find(':') != std::string::npos) {
      return false;
    }
#endif

    bool last_is_file = false;
    std::string segment;
    for (size_t i = 0; i <= rel_path.size(); ++i) {
      if (i == rel_path.size() || rel_path[i] == '/' || rel_path[i] == '\\') {
        if (segment.empty() || segment == "..") {
          return false;
        }
        if (segment == ".") {
          last_is_file = false;
        } else {
#ifdef _WIN32
          if (is_windows_reserved_segment(segment)) {
            return false;
          }
#endif
          last_is_file = true;
        }
        segment.clear();
      } else {
        segment.push_back(rel_path[i]);
      }
    }
    return last_is_file;
  }

  // Resolves rel_path relative to this directory and verifies that the canonical target
  // is an existing regular file strictly enclosed within the canonical directory.
  bool resolve_enclosed_regular_file(const std::string& rel_path, std::string& resolved_out) const {
    if (path_.empty() || path_.find('\0') != std::string::npos || !is_safe_relative_path(rel_path)) {
      return false;
    }

    const path candidate = join(rel_path);
#ifdef _WIN32
    if (w_path_.empty() || candidate.w_path_.empty()) {
      return false;
    }

    struct ScopedHandle {
      HANDLE h{INVALID_HANDLE_VALUE};
      ~ScopedHandle() {
        if (h != INVALID_HANDLE_VALUE) {
          CloseHandle(h);
        }
      }
    };

    ScopedHandle dir_handle{CreateFileW(w_path_.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                        nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr)};
    if (dir_handle.h == INVALID_HANDLE_VALUE) {
      return false;
    }
    BY_HANDLE_FILE_INFORMATION dir_info{};
    if (!GetFileInformationByHandle(dir_handle.h, &dir_info) ||
        (dir_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
      return false;
    }

    ScopedHandle file_handle{CreateFileW(candidate.w_path_.c_str(), GENERIC_READ,
                                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                         FILE_ATTRIBUTE_NORMAL, nullptr)};
    if (file_handle.h == INVALID_HANDLE_VALUE || GetFileType(file_handle.h) != FILE_TYPE_DISK) {
      return false;
    }
    BY_HANDLE_FILE_INFORMATION file_info{};
    if (!GetFileInformationByHandle(file_handle.h, &file_info) ||
        (file_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
      return false;
    }

    constexpr DWORD kFileNameNormalized = 0x0;
    constexpr DWORD kVolumeNameDos = 0x0;
    constexpr DWORD kVolumeNameNt = 0x2;

    std::wstring canon_dir;
    std::wstring canon_file;
    bool used_dos_path = get_final_path(dir_handle.h, kFileNameNormalized | kVolumeNameDos, canon_dir) &&
                         get_final_path(file_handle.h, kFileNameNormalized | kVolumeNameDos, canon_file);
    if (!used_dos_path) {
      // Restricted sandboxes (e.g., LPAC AppContainer) may block MountPointManager;
      // fall back to kernel NT volume paths so canonical containment is still enforced.
      if (!get_final_path(dir_handle.h, kFileNameNormalized | kVolumeNameNt, canon_dir) ||
          !get_final_path(file_handle.h, kFileNameNormalized | kVolumeNameNt, canon_file)) {
        return false;
      }
    }

    while (!canon_dir.empty() && canon_dir.back() == L'\\') {
      canon_dir.pop_back();
    }
    if (canon_dir.empty() || canon_file.size() <= canon_dir.size() + 1 || canon_file[canon_dir.size()] != L'\\' ||
        !equals_ordinal_ignore_case(canon_file.c_str(), canon_dir.c_str(), canon_dir.size())) {
      return false;
    }

    resolved_out = used_dos_path ? from_wstring(normalize_dos_final_path(canon_file)) : candidate.string();
    return !resolved_out.empty();
#else
    std::unique_ptr<char, decltype(&std::free)> raw_dir(realpath(path_.c_str(), nullptr), &std::free);
    std::unique_ptr<char, decltype(&std::free)> raw_file(realpath(candidate.path_.c_str(), nullptr), &std::free);
    if (!raw_dir || !raw_file) {
      return false;
    }

    std::string canon_dir(raw_dir.get());
    std::string canon_file(raw_file.get());
    while (canon_dir.size() > 1 && canon_dir.back() == '/') {
      canon_dir.pop_back();
    }

    const bool inside = (canon_dir == "/") ? (canon_file.size() > 1 && canon_file[0] == '/')
                                           : (canon_file.size() > canon_dir.size() + 1 &&
                                              canon_file.compare(0, canon_dir.size(), canon_dir) == 0 &&
                                              canon_file[canon_dir.size()] == '/');
    if (!inside) {
      return false;
    }

    struct stat st{};
    if (stat(canon_file.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
      return false;
    }

    resolved_out = std::move(canon_file);
    return true;
#endif
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
    if (path_.empty()) {
      return {};
    }
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, path_.data(), static_cast<int>(path_.size()), nullptr, 0);
    if (size_needed <= 0) {
      return {};
    }
    std::wstring utf16_str(size_needed, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, path_.data(), static_cast<int>(path_.size()), &utf16_str[0], size_needed);
    return utf16_str;
  }

  static std::string from_wstring(const std::wstring& wpath) {
    if (wpath.empty()) {
      return {};
    }
    int size_needed =
        WideCharToMultiByte(CP_UTF8, 0, wpath.data(), static_cast<int>(wpath.size()), nullptr, 0, nullptr, nullptr);
    if (size_needed <= 0) {
      return {};
    }
    std::string utf8_str(size_needed, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wpath.data(), static_cast<int>(wpath.size()), &utf8_str[0], size_needed, nullptr,
                        nullptr);
    return utf8_str;
  }

  static bool get_final_path(HANDLE handle, DWORD flags, std::wstring& out) {
    using GetFinalPathFn = DWORD(WINAPI*)(HANDLE, LPWSTR, DWORD, DWORD);
    static const GetFinalPathFn fn = []() -> GetFinalPathFn {
      HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
      return kernel32 ? reinterpret_cast<GetFinalPathFn>(GetProcAddress(kernel32, "GetFinalPathNameByHandleW"))
                      : nullptr;
    }();
    if (!fn) {
      return false;
    }
    DWORD needed = fn(handle, nullptr, 0, flags);
    if (needed == 0) {
      return false;
    }
    std::wstring buffer(needed, L'\0');
    DWORD written = fn(handle, &buffer[0], needed, flags);
    if (written == 0 || written >= needed) {
      return false;
    }
    buffer.resize(written);
    out = std::move(buffer);
    return true;
  }

  static bool equals_ordinal_ignore_case(const wchar_t* lhs, const wchar_t* rhs, size_t count) {
    using CompareOrdinalFn = int(WINAPI*)(LPCWCH, int, LPCWCH, int, BOOL);
    static const CompareOrdinalFn fn = []() -> CompareOrdinalFn {
      HMODULE kernel32 = GetModuleHandleW(L"kernel32.dll");
      return kernel32 ? reinterpret_cast<CompareOrdinalFn>(GetProcAddress(kernel32, "CompareStringOrdinal")) : nullptr;
    }();
    if (fn) {
      constexpr int kCstrEqual = 2;
      return fn(lhs, static_cast<int>(count), rhs, static_cast<int>(count), TRUE) == kCstrEqual;
    }
    return _wcsnicmp(lhs, rhs, count) == 0;
  }

  static std::wstring normalize_dos_final_path(const std::wstring& final_path) {
    if (final_path.compare(0, 8, L"\\\\?\\UNC\\") == 0 && final_path.size() - 6 < MAX_PATH) {
      return L"\\\\" + final_path.substr(8);
    }
    if (final_path.compare(0, 4, L"\\\\?\\") == 0 && final_path.size() >= 6 && final_path[5] == L':' &&
        final_path.size() - 4 < MAX_PATH) {
      return final_path.substr(4);
    }
    return final_path;
  }

  static bool is_windows_reserved_segment(const std::string& segment) {
    if (segment.back() == ' ' || segment.back() == '.') {
      return true;
    }
    const auto dot = segment.find('.');
    std::string stem = segment.substr(0, dot);
    for (char& ch : stem) {
      ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    }
    if (stem == "CON" || stem == "PRN" || stem == "AUX" || stem == "NUL" || stem == "CONIN$" || stem == "CONOUT$") {
      return true;
    }
    if (stem.size() == 4 && (stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0) && stem[3] >= '1' &&
        stem[3] <= '9') {
      return true;
    }
    if (stem.size() == 5 && (stem.compare(0, 3, "COM") == 0 || stem.compare(0, 3, "LPT") == 0) &&
        static_cast<unsigned char>(stem[3]) == 0xC2 &&
        (static_cast<unsigned char>(stem[4]) == 0xB9 || static_cast<unsigned char>(stem[4]) == 0xB2 ||
         static_cast<unsigned char>(stem[4]) == 0xB3)) {
      return true;
    }
    return false;
  }
#endif  // _WIN32
};

}  // namespace ort_extensions

namespace ortx = ort_extensions;

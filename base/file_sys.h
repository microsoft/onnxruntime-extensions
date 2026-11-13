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
#else
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#endif  // _WIN32

#include <sys/stat.h>

#include <cctype>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

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

  static constexpr const char* separators() {
#ifdef _WIN32
    return "\\/";
#else
    return "/";
#endif
  }

  static constexpr bool is_separator(char ch) {
#ifdef _WIN32
    return ch == '\\' || ch == '/';
#else
    return ch == '/';
#endif
  }

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
    if (is_separator(path_.back())) {
      return path(path_ + str);
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
    auto sep_pos = path_.find_last_of(separators());
    auto dot_pos = path_.find_last_of('.');
    if (dot_pos == std::string::npos || (sep_pos != std::string::npos && dot_pos < sep_pos)) {
      return "";
    }
    return path_.substr(dot_pos);
  }

  std::string parent_path() const {
    auto pos = path_.find_last_of(separators());
    if (pos == std::string::npos) {
      return "";
    }
    while (pos > 0 && is_separator(path_[pos - 1])) {
      --pos;
    }
    if (pos == 0) {
      return path_.substr(0, 1);
    }
#ifdef _WIN32
    if (pos == 2 && std::isalpha(static_cast<unsigned char>(path_[0])) && path_[1] == ':') {
      return path_.substr(0, 3);
    }
#endif  // _WIN32
    return path_.substr(0, pos);
  }

  // Validates lexically that rel_path is a relative file path without parent traversal,
  // drive/root prefixes, embedded NULs, or Windows device/stream syntax.
  static bool is_safe_relative_path(const std::string& rel_path) {
    if (rel_path.empty() || rel_path.find('\0') != std::string::npos) {
      return false;
    }
    if (is_separator(rel_path.front()) || is_separator(rel_path.back())) {
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
      if (i == rel_path.size() || is_separator(rel_path[i])) {
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

  // Opens rel_path relative to this directory without following escaping links, verifies
  // that the opened handle/descriptor is a regular file strictly enclosed within this
  // directory, and reads its bytes directly from that verified handle/descriptor.
  bool open_enclosed_regular_file(const std::string& rel_path,
                                  std::string& content_out,
                                  std::string& resolved_out) const {
    constexpr uint64_t kMaxEnclosedFileBytes = 512ULL * 1024ULL * 1024ULL;
    if (path_.empty() || path_.find('\0') != std::string::npos || !is_safe_relative_path(rel_path)) {
      return false;
    }

    std::vector<std::string> segments;
    std::string seg;
    for (size_t i = 0; i <= rel_path.size(); ++i) {
      if (i == rel_path.size() || is_separator(rel_path[i])) {
        if (!seg.empty() && seg != ".") {
          segments.push_back(seg);
        }
        seg.clear();
      } else {
        seg.push_back(rel_path[i]);
      }
    }
    if (segments.empty()) {
      return false;
    }

    const path candidate = join(rel_path);
#ifdef _WIN32
    if (w_path_.empty() || candidate.w_path_.empty()) {
      return false;
    }

    struct ScopedHandle {
      HANDLE h{INVALID_HANDLE_VALUE};
      ScopedHandle() = default;
      explicit ScopedHandle(HANDLE handle) : h(handle) {}
      ScopedHandle(const ScopedHandle&) = delete;
      ScopedHandle& operator=(const ScopedHandle&) = delete;
      ScopedHandle(ScopedHandle&& other) noexcept : h(other.h) { other.h = INVALID_HANDLE_VALUE; }
      ScopedHandle& operator=(ScopedHandle&& other) noexcept {
        if (this != &other) {
          if (h != INVALID_HANDLE_VALUE) {
            CloseHandle(h);
          }
          h = other.h;
          other.h = INVALID_HANDLE_VALUE;
        }
        return *this;
      }
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

    path cur = *this;
    std::vector<ScopedHandle> intermediate_handles;
    intermediate_handles.reserve(segments.size() - 1);
    for (size_t i = 0; i + 1 < segments.size(); ++i) {
      cur = cur.join(segments[i]);
      if (cur.w_path_.empty()) {
        return false;
      }
      ScopedHandle sub_handle{CreateFileW(cur.w_path_.c_str(), 0,
                                          FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                          OPEN_EXISTING,
                                          FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
      if (sub_handle.h == INVALID_HANDLE_VALUE) {
        return false;
      }
      BY_HANDLE_FILE_INFORMATION sub_info{};
      if (!GetFileInformationByHandle(sub_handle.h, &sub_info) ||
          (sub_info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0 ||
          (sub_info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
        return false;
      }
      intermediate_handles.push_back(std::move(sub_handle));
    }

    const path leaf_path = cur.join(segments.back());
    if (leaf_path.w_path_.empty()) {
      return false;
    }
    ScopedHandle file_handle{CreateFileW(leaf_path.w_path_.c_str(), GENERIC_READ,
                                         FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                                         FILE_FLAG_OPEN_REPARSE_POINT, nullptr)};
    if (file_handle.h == INVALID_HANDLE_VALUE || GetFileType(file_handle.h) != FILE_TYPE_DISK) {
      return false;
    }
    BY_HANDLE_FILE_INFORMATION file_info{};
    if (!GetFileInformationByHandle(file_handle.h, &file_info) ||
        (file_info.dwFileAttributes & (FILE_ATTRIBUTE_DIRECTORY | FILE_ATTRIBUTE_REPARSE_POINT)) != 0) {
      return false;
    }

    constexpr DWORD kFileNameNormalized = 0x0;
    constexpr DWORD kVolumeNameDos = 0x0;
    constexpr DWORD kVolumeNameNt = 0x2;

    std::wstring canon_dir;
    std::wstring canon_file;
    if (!(get_final_path(dir_handle.h, kFileNameNormalized | kVolumeNameDos, canon_dir) &&
          get_final_path(file_handle.h, kFileNameNormalized | kVolumeNameDos, canon_file))) {
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
    // GetFinalPathNameByHandleW with FILE_NAME_NORMALIZED returns exact on-disk entry casing
    // for shared ancestor directories, so exact ordinal comparison works on case-insensitive
    // volumes while rejecting case-sensitive sibling directories (e.g., tok vs TOK).
    if (canon_dir.empty() || canon_file.size() <= canon_dir.size() + 1 || canon_file[canon_dir.size()] != L'\\' ||
        canon_file.compare(0, canon_dir.size(), canon_dir) != 0) {
      return false;
    }

    LARGE_INTEGER file_size{};
    if (!GetFileSizeEx(file_handle.h, &file_size) || file_size.QuadPart < 0 ||
        static_cast<uint64_t>(file_size.QuadPart) > kMaxEnclosedFileBytes) {
      return false;
    }

    std::string data;
    data.reserve(static_cast<size_t>(file_size.QuadPart));
    char chunk[65536];
    for (;;) {
      DWORD bytes_read = 0;
      if (!ReadFile(file_handle.h, chunk, sizeof(chunk), &bytes_read, nullptr)) {
        return false;
      }
      if (bytes_read == 0) {
        break;
      }
      if (static_cast<uint64_t>(data.size()) + static_cast<uint64_t>(bytes_read) > kMaxEnclosedFileBytes) {
        return false;
      }
      data.append(chunk, bytes_read);
    }

    content_out = std::move(data);
    resolved_out = candidate.string();
    return true;
#else
    struct ScopedFd {
      int fd{-1};
      ~ScopedFd() {
        if (fd >= 0) {
          ::close(fd);
        }
      }
    };

#ifdef O_CLOEXEC
    constexpr int kCloExec = O_CLOEXEC;
#else
    constexpr int kCloExec = 0;
#endif
#ifdef O_NONBLOCK
    constexpr int kNonBlock = O_NONBLOCK;
#else
    constexpr int kNonBlock = 0;
#endif

    ScopedFd dir_fd{::open(path_.c_str(), O_RDONLY | O_DIRECTORY | kCloExec)};
    if (dir_fd.fd < 0) {
      return false;
    }

    for (size_t i = 0; i + 1 < segments.size(); ++i) {
      int next = ::openat(dir_fd.fd, segments[i].c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | kCloExec);
      if (next < 0) {
        return false;
      }
      ::close(dir_fd.fd);
      dir_fd.fd = next;
    }

    ScopedFd file_fd{::openat(dir_fd.fd, segments.back().c_str(), O_RDONLY | O_NOFOLLOW | kNonBlock | kCloExec)};
    if (file_fd.fd < 0) {
      return false;
    }

    struct stat st{};
    if (::fstat(file_fd.fd, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        static_cast<uint64_t>(st.st_size) > kMaxEnclosedFileBytes) {
      return false;
    }

#ifdef O_NONBLOCK
    int flags = ::fcntl(file_fd.fd, F_GETFL);
    if (flags >= 0 && (flags & O_NONBLOCK) != 0) {
      (void)::fcntl(file_fd.fd, F_SETFL, flags & ~O_NONBLOCK);
    }
#endif

    std::string data;
    data.reserve(static_cast<size_t>(st.st_size));
    char chunk[65536];
    for (;;) {
      ssize_t n = ::read(file_fd.fd, chunk, sizeof(chunk));
      if (n < 0) {
        if (errno == EINTR) {
          continue;
        }
        return false;
      }
      if (n == 0) {
        break;
      }
      if (static_cast<uint64_t>(data.size()) + static_cast<uint64_t>(n) > kMaxEnclosedFileBytes) {
        return false;
      }
      data.append(chunk, static_cast<size_t>(n));
    }

    content_out = std::move(data);
    resolved_out = candidate.string();
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

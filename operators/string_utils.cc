#ifdef ENABLE_TF_STRING
#include "farmhash.h"
#endif

#include "string_utils.h"

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

bool IsChineseChar(char32_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF)
         || (c >= 0x3400 && c <= 0x4DBF)
         || (c >= 0x20000 && c <= 0x2A6DF)
         || (c >= 0x2A700 && c <= 0x2B73F)
         || (c >= 0x2B740 && c <= 0x2B81F)
         || (c >= 0x2B820 && c <= 0x2CEAF)
         || (c >= 0xF900 && c <= 0xFAFF)
         || (c >= 0x2F800 && c <= 0x2FA1F);
}

bool IsAccent(char32_t c)
{
  // only support part of accent
  // [TODO] support more accent
  return c >= 0x300 && c <= 0x36F;
}

char32_t StripAccent(char32_t c)
{
  //   "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ"
  const char* tr = "AAAAAAÆCEEEEIIIIÐNOOOOO×ØUUUUYÞßaaaaaaæceeeeiiiiðnooooo÷øuuuuyþy";
  if (c < 192 || c > 255) {
    return c;
}

  return tr[c - 192];
}

#ifdef ENABLE_TF_STRING
// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hash.cc#L28
static inline uint64_t ByteAs64(char c) { return static_cast<uint64_t>(c) & 0xff; }

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/raw_coding.h#L41
uint64_t DecodeFixed32(const char* ptr) {
  return ((static_cast<uint64_t>(static_cast<unsigned char>(ptr[0]))) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[1])) << 8) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[2])) << 16) |
          (static_cast<uint64_t>(static_cast<unsigned char>(ptr[3])) << 24));
}

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/raw_coding.h#L55
static uint64_t DecodeFixed64(const char* ptr) {
  uint64_t lo = DecodeFixed32(ptr);
  uint64_t hi = DecodeFixed32(ptr + 4);
  return (hi << 32) | lo;
}

// Source: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/platform/hash.cc#L79
uint64_t Hash64(const char* data, size_t n, uint64_t seed) {
  const uint64_t m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64_t h = seed ^ (n * m);

  while (n >= 8) {
    uint64_t k = DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

uint64_t Hash64Fast(const char* data, size_t n) {
  return static_cast<int64_t>(util::Fingerprint64(data, n));
}

#endif  // ENABLE_TF_STRING

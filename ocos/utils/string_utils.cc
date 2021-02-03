#include "string_utils.h"

std::vector<std::string_view> SplitString(const std::string_view& str, const std::string_view& seps, bool remove_empty_entries) {
  std::vector<std::string_view> result;
  std::string ::size_type pre_pos = 0;

  while (true) {
    auto next_pos = str.find_first_of(seps, pre_pos);
    //TODO: need to merge empty check with pos check
    if (pre_pos != next_pos || !remove_empty_entries) {
      auto token_len = next_pos == std::string::npos ? std::string::npos : next_pos - pre_pos;

      auto sub_str = str.substr(pre_pos, token_len);

      if (!sub_str.empty()) {
        result.push_back(sub_str);
      }
    }

    if (next_pos == std::string::npos) {
      break;
    }

    pre_pos = next_pos + 1;
  }

  return result;
}

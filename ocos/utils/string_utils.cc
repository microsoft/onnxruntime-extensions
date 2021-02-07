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

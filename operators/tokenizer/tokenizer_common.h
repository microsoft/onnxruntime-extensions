// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>

#include "ortx_tokenizer.h"
#include "ext_status.h"
#include "op_def_struct.h"
#include "nlohmann/json_fwd.hpp"

#include "ustring.h"


namespace ort_extensions {
class BpeModel;

struct AddedToken final {
  uint32_t id_{};
  std::string token_type_;
  std::string content_;
  bool lstrip_{};
  bool normalized_{};
  bool rstrip_{};
  bool single_word_{};
  bool special_{};
};

class TokenJsonConfig;  // forward declaration

struct TokenizerDecodingState {
  bool f_special_last{};
  std::string incomplete_utf8_;
};

using AddedTokenMap = std::unordered_map<std::u32string, AddedToken>;

constexpr std::string_view spm_escaped_space = "\xE2\x96\x81";
}  // namespace ort_extensions

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

namespace ort_extensions {
class BpeModel;

namespace bpe {

struct AddedToken final {
  uint32_t id_{};
  std::string token_type_;
  std::string content_;
  bool lstrip_{};
  bool normalized_{};
  bool rstrip_{};
  bool single_word_{};
};

class TokenJsonConfig;  // forward declaration

}  // namespace bpe
}  // namespace ort_extensions

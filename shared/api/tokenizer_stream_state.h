#pragma once

#include <memory>
#include <string>

#include "c_api_utils.hpp"
#include "tokenizer_common.h"

namespace ort_extensions {

class DetokenizerCache : public OrtxObjectImpl {
 public:
  DetokenizerCache();
  ~DetokenizerCache() override;

  std::unique_ptr<TokenizerDecodingState> decoder_state_;
  std::string last_text_;
};

}  // namespace ort_extensions
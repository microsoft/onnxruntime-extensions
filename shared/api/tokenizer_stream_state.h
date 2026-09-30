#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>

#include "c_api_utils.hpp"
#include "tokenizer_common.h"
#include "tokenizer_word_grouping.h"

namespace ort_extensions {

enum class DetokenizerCacheMode {
  Unset,
  Text,
  Metadata,
};

class DetokenizerCache : public OrtxObjectImpl {
 public:
  DetokenizerCache();
  ~DetokenizerCache() override;

  extError_t SetMode(DetokenizerCacheMode requested_mode);
  extError_t ConfigureMetadata(const OrtxMetadataConfig& config);
  bool HasTimestampTrackingSetting() const { return track_timestamp_metadata_.has_value(); }
  void ConfigureTimestampTracking(bool enabled);
  bool TracksTimestamps() const { return track_timestamp_metadata_.value_or(false); }
  void ConsumeTimestamp(const TokenizerWordPieceInfo& piece, std::string_view text);
  void FinalizeMetadata();
  const OrtxMetadata& Metadata() const { return metadata_; }

  std::unique_ptr<TokenizerDecodingState> decoder_state_;
  std::string last_text_;

 private:
  struct TimestampState;
  std::unique_ptr<TimestampState> timestamp_state_;
  std::optional<bool> track_timestamp_metadata_;
  OrtxMetadata metadata_{};
  DetokenizerCacheMode mode_{DetokenizerCacheMode::Unset};
};

}  // namespace ort_extensions
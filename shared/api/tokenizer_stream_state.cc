#include "tokenizer_stream_state.h"

namespace ort_extensions {

struct DetokenizerCache::TimestampState {
  TokenizerWordGroupingState grouping;
  std::vector<OrtxTimestampWordMetadata> word_views;
  OrtxTimestampMetadata metadata{};

  void UpdateTimestampMetadataView() {
    word_views.clear();
    word_views.reserve(grouping.CompletedWords().size());
    for (const auto& word : grouping.CompletedWords()) {
      word_views.push_back({word.text.c_str(), word.start_token_index, word.stop_token_index});
    }
    metadata = {word_views.data(), word_views.size(), grouping.FirstPendingTokenIndex()};
  }
};

DetokenizerCache::DetokenizerCache() : OrtxObjectImpl(kOrtxKindDetokenizerCache) {}
DetokenizerCache::~DetokenizerCache() = default;

extError_t DetokenizerCache::SetMode(DetokenizerCacheMode requested_mode) {
  if (mode_ == DetokenizerCacheMode::Unset) {
    mode_ = requested_mode;
    return kOrtxOK;
  }
  if (mode_ == requested_mode) return kOrtxOK;

  ReturnableStatus::last_error_message_ =
      "Cannot mix OrtxDetokenizeCached and OrtxDetokenizeCachedWithMetadata on the same cache. "
      "Destroy and recreate the detokenizer cache to switch modes.";
  return kOrtxErrorInvalidArgument;
}

void DetokenizerCache::ConfigureTimestampTracking(bool enabled) {
  if (!track_timestamp_metadata_.has_value()) track_timestamp_metadata_ = enabled;
}

void DetokenizerCache::ConsumeTimestamp(const TokenizerWordPieceInfo& piece, std::string_view text) {
  if (!TracksTimestamps()) return;
  if (!timestamp_state_) timestamp_state_ = std::make_unique<TimestampState>();
  timestamp_state_->grouping.Consume(piece, text);
  timestamp_state_->UpdateTimestampMetadataView();
  metadata_.timestampMetadata = &timestamp_state_->metadata;
}

void DetokenizerCache::FinalizeMetadata() {
  if (metadata_.timestampMetadata) {
    timestamp_state_->grouping.Finalize();
    timestamp_state_->UpdateTimestampMetadataView();
  }
}

}  // namespace ort_extensions
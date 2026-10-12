// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "tokenizer_common.h"

namespace ort_extensions {

struct CompletedTokenizerWord {
  std::string text;
  size_t start_token_index{};
  size_t stop_token_index{};
};

class TokenizerWordGroupingState {
 public:
  void Consume(const TokenizerWordPieceInfo& piece, std::string_view decoded_text);
  void Finalize();
  const std::vector<CompletedTokenizerWord>& CompletedWords() const;
  size_t FirstPendingTokenIndex() const;

 private:
  struct PendingText {
    std::string text;
    size_t start_token_index{};
    size_t stop_token_index{};
    bool active{};
  };

  void Append(PendingText& pending, std::string_view text, size_t start_token_index, size_t stop_token_index);
  void CompleteWord();
  void ConsumeTextPart(std::string_view text, size_t start_token_index, size_t stop_token_index, bool starts_word);

  size_t next_token_index_{};
  PendingText pending_word_;
  PendingText pending_delimiter_;
  std::optional<size_t> buffered_output_start_token_index_;
  std::vector<CompletedTokenizerWord> completed_words_;
};

class TokenizerDiarization {
 public:
  static int AggregateWordSpeaker(const double* start_sums, const double* stop_sums, int64_t start_count,
                                  int64_t stop_count);
  static bool SpeakerChanged(bool has_pending, int pending_speaker, int speaker);
};

bool SpeechSegmentStartsNew(bool has_pending, bool diarization, int pending_speaker, int speaker, int64_t start,
                            int64_t previous_stop, int64_t gap);

bool SpeechSegmentEnds(std::string_view text, const char* const* separators, size_t count);

}  // namespace ort_extensions

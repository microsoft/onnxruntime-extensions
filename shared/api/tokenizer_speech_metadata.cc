// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_speech_metadata.h"
#include "string_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ort_extensions {
namespace {

bool IsWhitespace(char32_t value) { return IsSpace(value) || value == U'\v' || value == U'\f'; }

bool IsWhitespace(std::string_view text) {
  const ustring codepoints{text};
  return !codepoints.empty() &&
         std::all_of(codepoints.begin(), codepoints.end(), [](char32_t value) { return IsWhitespace(value); });
}

bool IsPunctuation(std::string_view text) {
  bool found = false;
  for (const auto value : ustring(text)) {
    if (IsWhitespace(value)) continue;
    if (!IsPunct(value)) return false;
    found = true;
  }
  return found;
}

}  // namespace

void TokenizerWordGroupingState::Append(PendingText& pending, std::string_view text, size_t start_token_index,
                                        size_t stop_token_index) {
  if (!pending.active) {
    pending.start_token_index = start_token_index;
    pending.active = true;
  }
  pending.text.append(text);
  pending.stop_token_index = stop_token_index;
}

void TokenizerWordGroupingState::CompleteWord() {
  if (!pending_word_.active || pending_word_.text.empty()) return;
  completed_words_.push_back(
      {std::move(pending_word_.text), pending_word_.start_token_index, pending_word_.stop_token_index});
  pending_word_ = {};
}

void TokenizerWordGroupingState::ConsumeTextPart(std::string_view text, size_t start_token_index,
                                                 size_t stop_token_index, bool starts_word) {
  if (IsWhitespace(text)) {
    Append(pending_delimiter_, text, start_token_index, stop_token_index);
    return;
  }

  const bool punctuation = IsPunctuation(text);
  if ((starts_word || pending_delimiter_.active) && !punctuation) {
    CompleteWord();
  }

  if (pending_delimiter_.active) {
    Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index,
           pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  Append(pending_word_, text, start_token_index, stop_token_index);
}

void TokenizerWordGroupingState::Consume(const TokenizerWordPieceInfo& piece, std::string_view decoded_text) {
  completed_words_.clear();
  const size_t token_index = next_token_index_++;
  if (decoded_text.empty()) {
    if (!piece.is_special && !buffered_output_start_token_index_) {
      buffered_output_start_token_index_ = token_index;
    }
    return;
  }

  const size_t output_start_token_index = buffered_output_start_token_index_.value_or(token_index);
  buffered_output_start_token_index_.reset();

  const ustring codepoints{decoded_text};
  size_t offset = 0;
  for (size_t index = 0; index < codepoints.size();) {
    const bool first_part = index == 0;
    const bool whitespace = IsWhitespace(codepoints[index]);
    size_t stop = offset;
    do {
      size_t width = ustring::UTF8Len(decoded_text[stop]);
      if (width > decoded_text.size() - stop || static_cast<unsigned char>(decoded_text[stop]) >= 0xF8) width = 1;
      stop += width;
      ++index;
    } while (index < codepoints.size() && IsWhitespace(codepoints[index]) == whitespace);
    const bool starts_word = first_part && piece.starts_word;
    ConsumeTextPart(decoded_text.substr(offset, stop - offset), output_start_token_index, token_index + 1, starts_word);
    offset = stop;
  }

  if (piece.boundary_style == WordBoundaryStyle::SuffixBpe && piece.ends_word) {
    if (pending_delimiter_.active) {
      Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index,
             pending_delimiter_.stop_token_index);
      pending_delimiter_ = {};
    }
    CompleteWord();
  }
}

void TokenizerWordGroupingState::Finalize() {
  completed_words_.clear();
  buffered_output_start_token_index_.reset();
  if (pending_delimiter_.active) {
    Append(pending_word_, pending_delimiter_.text, pending_delimiter_.start_token_index,
           pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  CompleteWord();
}

const std::vector<CompletedTokenizerWord>& TokenizerWordGroupingState::CompletedWords() const {
  return completed_words_;
}

size_t TokenizerWordGroupingState::FirstPendingTokenIndex() const {
  size_t first_pending = next_token_index_;
  if (pending_word_.active) first_pending = std::min(first_pending, pending_word_.start_token_index);
  if (pending_delimiter_.active) first_pending = std::min(first_pending, pending_delimiter_.start_token_index);
  if (buffered_output_start_token_index_) {
    first_pending = std::min(first_pending, *buffered_output_start_token_index_);
  }
  return first_pending;
}

void TokenizerDiarization::ValidateSpeakerEvidence(const double* start_sums, const double* stop_sums,
                                                   int64_t start_count, int64_t stop_count) {
  if (!start_sums || !stop_sums) throw std::runtime_error("Speaker probability sums must not be null");
  if (start_count < 0 || stop_count <= start_count)
    throw std::runtime_error("Speaker evidence requires a positive evaluated-frame count");
  for (size_t speaker = 0; speaker < 8; ++speaker) {
    const double sum = stop_sums[speaker] - start_sums[speaker];
    const double count = static_cast<double>(stop_count - start_count);
    if (!std::isfinite(start_sums[speaker]) || !std::isfinite(stop_sums[speaker]) || start_sums[speaker] < 0 ||
        stop_sums[speaker] < 0 || start_sums[speaker] > static_cast<double>(start_count) + 1e-9 ||
        stop_sums[speaker] > static_cast<double>(stop_count) + 1e-9 || sum < -1e-9 || sum > count + 1e-9)
      throw std::runtime_error("Speaker evidence contains invalid probability sums");
  }
}

int TokenizerDiarization::AggregateWordSpeaker(const double* start_sums, const double* stop_sums, int64_t start_count,
                                               int64_t stop_count) {
  ValidateSpeakerEvidence(start_sums, stop_sums, start_count, stop_count);

  double scores[8];
  double total = 0;
  for (size_t speaker = 0; speaker < 8; ++speaker) {
    const double sum = stop_sums[speaker] - start_sums[speaker];
    const double count = static_cast<double>(stop_count - start_count);
    scores[speaker] = std::clamp(sum / count, 0.01, 1.0);
    total += scores[speaker];
  }
  for (double& score : scores) score /= total;
  return static_cast<int>(std::max_element(scores, scores + 8) - scores);
}

bool TokenizerDiarization::SpeakerChanged(bool has_pending, int pending_speaker, int speaker) {
  return has_pending && pending_speaker != speaker;
}

bool SpeechSegmentStartsNew(bool has_pending, bool diarization, int pending_speaker, int speaker, int64_t start,
                            int64_t previous_stop, int64_t gap) {
  if (!has_pending) return false;
  if (diarization) return TokenizerDiarization::SpeakerChanged(has_pending, pending_speaker, speaker);
  return gap >= 0 && (gap == 0 || start - previous_stop >= gap);
}

bool SpeechSegmentEnds(std::string_view text, const char* const* separators, size_t count) {
  if (count && !separators) throw std::runtime_error("Segment separators must not be null");
  while (!text.empty() && (text.back() == ' ' || text.back() == '\t' || text.back() == '\n' || text.back() == '\r' ||
                           text.back() == '\f' || text.back() == '\v'))
    text.remove_suffix(1);
  for (size_t i = 0; i < count; ++i) {
    if (!separators[i]) throw std::runtime_error("Segment separator must not be null");
    const std::string_view separator{separators[i]};
    if (!separator.empty() && text.size() >= separator.size() &&
        text.substr(text.size() - separator.size()) == separator)
      return true;
  }
  return false;
}

}  // namespace ort_extensions
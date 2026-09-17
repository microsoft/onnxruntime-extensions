// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "tokenizer_word_grouping.h"

#include <algorithm>
#include <cctype>

namespace ort_extensions {
namespace {

bool IsWhitespace(std::string_view text) {
  return !text.empty() && std::all_of(text.begin(), text.end(), [](unsigned char value) {
           return std::isspace(value) != 0;
         });
}

bool IsPunctuation(std::string_view text) {
  bool found = false;
  for (const unsigned char value : text) {
    if (std::isspace(value)) continue;
    if (value > 0x7f || std::ispunct(value) == 0) return false;
    found = true;
  }
  return found;
}

}  // namespace

void TokenizerWordGroupingState::Append(PendingText& pending, std::string_view text,
                                        size_t start_token_index, size_t stop_token_index) {
  if (!pending.active) {
    pending.start_token_index = start_token_index;
    pending.active = true;
  }
  pending.text.append(text);
  pending.stop_token_index = stop_token_index;
}

void TokenizerWordGroupingState::CompleteWord() {
  if (!pending_word_.active || pending_word_.text.empty()) return;
  completed_words_.push_back({std::move(pending_word_.text),
                              pending_word_.start_token_index,
                              pending_word_.stop_token_index});
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
    Append(pending_word_, pending_delimiter_.text,
           pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  Append(pending_word_, text, start_token_index, stop_token_index);
}

void TokenizerWordGroupingState::Consume(const TokenizerWordPieceInfo& piece,
                                         std::string_view decoded_text) {
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

  bool first_part = true;
  for (size_t offset = 0; offset < decoded_text.size();) {
    const bool whitespace = std::isspace(static_cast<unsigned char>(decoded_text[offset])) != 0;
    size_t stop = offset + 1;
    while (stop < decoded_text.size() &&
           (std::isspace(static_cast<unsigned char>(decoded_text[stop])) != 0) == whitespace) {
      ++stop;
    }
    const bool starts_word = first_part && piece.boundary_style != WordBoundaryStyle::SuffixBpe &&
                             piece.encoded_piece != decoded_text;
    ConsumeTextPart(decoded_text.substr(offset, stop - offset), output_start_token_index,
            token_index + 1, starts_word);
    first_part = false;
    offset = stop;
  }

  if (piece.boundary_style == WordBoundaryStyle::SuffixBpe && piece.ends_word) {
    if (pending_delimiter_.active) {
      Append(pending_word_, pending_delimiter_.text,
             pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
      pending_delimiter_ = {};
    }
    CompleteWord();
  }
}

void TokenizerWordGroupingState::Finalize() {
  completed_words_.clear();
  if (pending_delimiter_.active) {
    Append(pending_word_, pending_delimiter_.text,
           pending_delimiter_.start_token_index, pending_delimiter_.stop_token_index);
    pending_delimiter_ = {};
  }
  CompleteWord();
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

}  // namespace ort_extensions
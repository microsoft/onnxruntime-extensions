// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ort_extensions {

enum class WordBoundaryStyle {
  SentencePiece,
  PrefixBpe,
  SuffixBpe,
};

struct TokenizerWordPieceInfo {
  std::string_view encoded_piece;
  WordBoundaryStyle boundary_style{WordBoundaryStyle::SentencePiece};
  bool is_special{};
  bool ends_word{};
};

struct CompletedTokenizerWord {
  std::string text;
  size_t start_token_index{};
  size_t stop_token_index{};
};

class TokenizerWordGroupingState {
 public:
  void Consume(const TokenizerWordPieceInfo& piece, std::string_view decoded_text);
  void Finalize();
  const std::vector<CompletedTokenizerWord>& CompletedWords() const { return completed_words_; }
  size_t FirstPendingTokenIndex() const;

 private:
  struct PendingText {
    std::string text;
    size_t start_token_index{};
    size_t stop_token_index{};
    bool active{};
  };

  void Append(PendingText& pending, std::string_view text, size_t start_token_index,
              size_t stop_token_index);
  void CompleteWord();
  void ConsumeTextPart(std::string_view text, size_t start_token_index,
                       size_t stop_token_index, bool starts_word);

  size_t next_token_index_{};
  PendingText pending_word_;
  PendingText pending_delimiter_;
  std::optional<size_t> buffered_output_start_token_index_;
  std::vector<CompletedTokenizerWord> completed_words_;
};

}  // namespace ort_extensions
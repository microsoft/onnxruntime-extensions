// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ustring.h"

#include <list>
#include <string>
#include <vector>
#include <functional>

#include "nlohmann/json.hpp"
#include "token_fwd.h"
#include "trietree.hpp"
#include "bpe_jsoncfg.hpp"

namespace ort_extensions {

struct SpmUgmTokenizer: public TokenizerKernelBase {
  using json = nlohmann::json;

  SpmUgmTokenizer() = default;
  OrtxStatus Load(const bpe::TokenJsonConfig& config) {
    // auto vocab = config["vocab"];
    // if (vocab.precompiled_charsmap.size() > 0) {
    //   size_t charsmap_offset = 0;

    //   // First four bytes of precompiled_charsmap contains length of binary
    //   // blob containing XOR-compressed compact double array (XCDA) entries
    //   uint32_t xcda_blob_size = *(const uint32_t*)&vocab.precompiled_charsmap[0];
    //   charsmap_offset += sizeof(xcda_blob_size);
    //   if (xcda_blob_size + charsmap_offset >= vocab.precompiled_charsmap.size()) {
    //     throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
    //   }

    //   // Next xcda_blob_size bytes contain entries of XOR-compressed compact
    //   // double array (XCDA). Each entry is bit-packed into a 32-bit integer.
    //   xcda_array = (const uint32_t*)&vocab.precompiled_charsmap[charsmap_offset];
    //   xcda_array_size = xcda_blob_size / sizeof(uint32_t);
    //   charsmap_offset += xcda_blob_size;

    //   // Remaining bytes of precompiled charsmap contain null-terminated
    //   // replacement strings for prefixes matched by the XCDA.
    //   prefix_replacements = &vocab.precompiled_charsmap[charsmap_offset];
    //   prefix_replacements_size = vocab.precompiled_charsmap.size() - charsmap_offset;
    // }

    // for (unsigned int id = 0; id < vocab.id_to_token.size(); ++id) {
    //   const auto& token_data = vocab.id_to_token[id];

    //   if (llama_is_normal_token(vocab, id)) {
    //     min_score = std::min<float>(min_score, token_data.score);
    //     max_score = std::max<float>(max_score, token_data.score);
    //   }

    //   if (llama_is_normal_token(vocab, id) || llama_is_user_defined_token(vocab, id) ||
    //       llama_is_unused_token(vocab, id)) {
    //     token_matcher.insert(token_data.text.data(), token_data.text.size(), id);
    //   }

    //   if (llama_is_user_defined_token(vocab, id)) {
    //     user_defined_token_matcher.insert(token_data.text.data(), token_data.text.size());
    //   }
    // }

    // unknown_token_score = min_score - unknown_token_score_penalty;
    return {};
  }

  extTokenId_t GetTokenId(const std::string& token) const {
    return 0;
  }

  OrtxStatus Compute(const ortc::Tensor<std::string>& input,
                     ortc::Tensor<int64_t>& tokenize_output) const {
    
    if (input.Shape().size() != 1) {
      return OrtxStatus(extError_t::kOrtxErrorInvalidArgument, "Input tensor must have rank 1.");
    }

    // // get current size of output (for reversal later)
    // size_t output_size = output.size();

    // normalize the input first
    std::string normalized;
    // Normalize(input.AsScalar(), &normalized);
    size_t input_len = normalized.size();
    if (input_len == 0) {
      return {};
    }

    // // initialize score_sum to -FLT_MAX so it will be always lower than sums of token scores
    // std::vector<struct best_tokenization> tokenization_results(input_len + 1, {vocab.special_unk_id, 0, -FLT_MAX});
    // // at the beginning tokenization score is zero
    // tokenization_results[0] = {vocab.special_unk_id, 0, 0};

    // for (size_t input_offset = 0; input_offset < input_len;) {
    //   size_t prefix_offset = input_offset;
    //   // calculate how many code units are in the currently processed UTF code point
    //   size_t n_utf8_code_units = std::min<size_t>(unicode_len_utf8(normalized[input_offset]), input_len - input_offset);

    //   // traverse the token matcher trie to find a matching token
    //   bool single_codepoint_token_found = false;
    //   const struct best_tokenization& current_best = tokenization_results[input_offset];
    //   const struct naive_trie* node = token_matcher.traverse(normalized[prefix_offset++]);

    //   while (prefix_offset <= input_len && node != NULL) {
    //     // check if we found valid token in prefix
    //     if (node->has_value) {
    //       // check if it corresponds to the whole UTF code point
    //       if (prefix_offset - input_offset == n_utf8_code_units) {
    //         single_codepoint_token_found = true;
    //       }
    //       llama_token token_id = node->value;
    //       const auto& token_data = vocab.id_to_token[token_id];

    //       // we set the user-defined token scores to 0 to make them more likely to be selected
    //       // (normal token scores are log probabilities, so they are negative)
    //       // score type is double here to make tokenization results exactly
    //       // the same as in the HF tokenizer using SentencePiece
    //       const double token_score = llama_is_user_defined_token(vocab, token_id) ? 0.0 : token_data.score;
    //       const double challenger_score = current_best.score_sum + token_score;
    //       struct best_tokenization& current_champ = tokenization_results[prefix_offset];
    //       if (challenger_score > current_champ.score_sum) {
    //         struct best_tokenization challenger = {token_id, input_offset, (float)challenger_score};
    //         current_champ = challenger;
    //       }
    //     }
    //     node = node->traverse(normalized[prefix_offset++]);
    //   }

    //   // if we didn't find a valid token corresponding to the whole UTF code point
    //   // then use unknown token as the tokenization of this UTF code point
    //   if (!single_codepoint_token_found) {
    //     const double challenger_score = current_best.score_sum + unknown_token_score;
    //     prefix_offset = input_offset + n_utf8_code_units;
    //     struct best_tokenization& current_champ = tokenization_results[prefix_offset];
    //     if (challenger_score > current_champ.score_sum) {
    //       struct best_tokenization challenger = {vocab.special_unk_id, input_offset, (float)challenger_score};
    //       current_champ = challenger;
    //     }
    //   }

    //   // move to the next UTF code point
    //   input_offset += n_utf8_code_units;
    // }

    // // now backtrack from the end to gather token ids of the best tokenization
    // // merge sequences of consecutive unknown tokens into single unknown tokens
    // bool is_prev_unknown = false;
    // for (struct best_tokenization& tokenization = tokenization_results[input_len];;
    //      tokenization = tokenization_results[tokenization.input_offset]) {
    //   bool is_unknown = tokenization.token_id == vocab.special_unk_id;
    //   if (!(is_prev_unknown && is_unknown)) {
    //     output.push_back(tokenization.token_id);
    //   }
    //   if (tokenization.input_offset == 0) {
    //     break;
    //   }
    //   is_prev_unknown = is_unknown;
    // }

    // // reverse the output since we added tokens starting from the end of the input
    // std::reverse(output.begin() + output_size, output.end());
    return {};
  }

 private:
  // helper structure for returning normalization results
  struct normalization_result {
    const char* normalized;
    size_t normalized_len;
    size_t consumed_input;
  };

  // void Normalize(const std::string& input, std::string* normalized) {
  //   normalized->clear();
  //   normalized->reserve(input.size() * 3);

  //   const std::string space = vocab.tokenizer_escape_whitespaces ? escaped_space : " ";

  //   bool shall_prepend_space = !vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
  //   bool shall_append_space = vocab.tokenizer_treat_whitespace_as_suffix && vocab.tokenizer_add_space_prefix;
  //   bool shall_merge_spaces = vocab.tokenizer_remove_extra_whitespaces;

  //   bool is_space_prepended = false;
  //   bool processing_non_ws = false;

  //   size_t input_len = input.size();

  //   for (size_t input_offset = 0; input_offset < input_len;) {
  //     auto norm_res = normalize_prefix(input, input_offset);
  //     for (size_t i = 0; i < norm_res.normalized_len; i++) {
  //       char c = norm_res.normalized[i];
  //       if (c != ' ') {
  //         if (!processing_non_ws) {
  //           processing_non_ws = true;
  //           if ((shall_prepend_space && !is_space_prepended) || shall_merge_spaces) {
  //             normalized->append(space);
  //             is_space_prepended = true;
  //           }
  //         }
  //         normalized->push_back(c);
  //       } else {
  //         if (processing_non_ws) {
  //           processing_non_ws = false;
  //         }
  //         if (!shall_merge_spaces) {
  //           normalized->append(space);
  //         }
  //       }
  //     }

  //     input_offset += norm_res.consumed_input;
  //   }

  //   if (shall_append_space) {
  //     normalized->append(space);
  //   }
  // }

  /*
   * This structure is a view wrapper for XOR-compressed double array (XCDA)
   * See Shunsuke Kanda (2018). Space- and Time-Efficient String Dictionaries.
   * Each bit-packed entry contains:
   * - BASE array value in bits 10-30
   * - LCHECK array value in bits 0-7
   * - LEAF array value in bit 9
   * Entries containing indexes of replacement sequences have set bit 31
   */
  struct xcda_array_view {
   public:
    xcda_array_view(const uint32_t* xcda_array, size_t xcda_array_size)
        : xcda_array(xcda_array), xcda_array_size(xcda_array_size) {}
    uint32_t get_base(size_t index) {
      uint32_t packed_node = get_node(index);
      return (packed_node >> 10) << ((packed_node & (1U << 9)) >> 6);
    }
    uint32_t get_lcheck(size_t index) {
      uint32_t packed_node = get_node(index);
      return packed_node & ((1U << 31) | 0xff);
    }
    bool get_leaf(size_t index) {
      uint32_t packed_node = get_node(index);
      return (packed_node >> 8) & 1;
    }
    uint32_t get_value(size_t index) {
      uint32_t packed_node = get_node(index);
      return packed_node & ((1U << 31) - 1);
    }

   private:
    uint32_t get_node(size_t index) {
      if (index > xcda_array_size) {
        throw std::runtime_error("Index out of array bounds in XCDA array!");
      }
      return xcda_array[index];
    }
    const uint32_t* xcda_array;
    size_t xcda_array_size;
  };

  // struct normalization_result normalize_prefix(const std::string& input, size_t input_offset) {
  //   if (input_offset == input.size()) {
  //     return {&input[input_offset], 0, 0};
  //   }

  //   // if input prefix matches some user-defined token return this token as normalization result
  //   auto user_defined_token_match =
  //       user_defined_token_matcher.get_longest_prefix(&input[input_offset], input.size() - input_offset);
  //   if (user_defined_token_match.second > 0) {
  //     return {&input[input_offset], user_defined_token_match.second, user_defined_token_match.second};
  //   }

  //   size_t longest_prefix_length = 0;
  //   size_t longest_prefix_offset = 0;

  //   if (xcda_array_size > 0) {
  //     struct xcda_array_view xcda_view(xcda_array, xcda_array_size);

  //     // Find the longest normalized sequence matching the input prefix by walking
  //     // the XOR-compressed compact double array (XCDA) starting from the root node
  //     // We find the index of the next node by calculating BASE[s] ^ c where s is
  //     // the index of the previous node and c is a numerical character value
  //     uint32_t node_index = 0;
  //     // get BASE of the root node
  //     node_index = xcda_view.get_base(node_index);
  //     for (size_t prefix_offset = input_offset; prefix_offset < input.size(); prefix_offset++) {
  //       unsigned char c = input[prefix_offset];
  //       if (c == 0) {
  //         break;
  //       }
  //       node_index ^= c;
  //       // if value of LCHECK is not c it means that this is not a child of
  //       // the previous node, so we stop matching
  //       if (xcda_view.get_lcheck(node_index) != c) {
  //         break;
  //       }
  //       bool is_leaf = xcda_view.get_leaf(node_index);
  //       // get BASE of the current node
  //       node_index ^= xcda_view.get_base(node_index);
  //       // if LEAF of the current node is true, it means that its BASE points to the node
  //       // containing index of replacement sequence for currently matched input prefix
  //       if (is_leaf) {
  //         longest_prefix_length = prefix_offset - input_offset + 1;
  //         // get index of replacement sequence for currently matched input prefix
  //         longest_prefix_offset = xcda_view.get_value(node_index);
  //       }
  //     }
  //   }

  //   if (longest_prefix_length > 0) {
  //     // we have a match, so return the replacement sequence
  //     if (longest_prefix_offset >= prefix_replacements_size) {
  //       throw std::runtime_error("Index out of array bounds in precompiled charsmap!");
  //     }
  //     const char* prefix_replacement = &prefix_replacements[longest_prefix_offset];
  //     return {prefix_replacement, strlen(prefix_replacement), longest_prefix_length};
  //   } else {
  //     // check if the input prefix contains a valid sequence of UTF-8 code units
  //     try {
  //       // if yes, return this sequence unmodified
  //       size_t prefix_offset = input_offset;
  //       unicode_cpt_from_utf8(input, prefix_offset);
  //       return {&input[input_offset], prefix_offset - input_offset, prefix_offset - input_offset};
  //     } catch (std::invalid_argument& /*ex*/) {
  //       // if no, consume 1 byte and return U+FFFD - REPLACEMENT CHARACTER
  //       return {"\xEF\xBF\xBD", 3, 1};
  //     }
  //   }
  // }

  // escaped space symbol - U+2581 (Lower One Eighth Block)
  const std::string escaped_space = "\xE2\x96\x81";

  const char* prefix_replacements = NULL;
  size_t prefix_replacements_size = 0;

  const uint32_t* xcda_array = NULL;
  size_t xcda_array_size = 0;

  TrieTree<char32_t> user_defined_token_matcher;

  // this structure stores the best tokenization so far at input_offset
  struct best_tokenization {
    extTokenId_t token_id;
    size_t input_offset;
    float score_sum;
  };

  float min_score = FLT_MAX;
  float max_score = -FLT_MAX;

  float unknown_token_score_penalty = 10.0;
  float unknown_token_score;

  TrieTree<char32_t> token_matcher;
};

} // namespace ort_extensions

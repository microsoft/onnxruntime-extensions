// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef ENABLE_GPT2_TOKENIZER
#include "bpe_tokenizer.hpp"
#include "bpe_kernels.h"
#include "bpe_decoder.hpp"
#endif

#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.hpp"
#include "sentencepiece_decoder.hpp"
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
#include "wordpiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BLINGFIRE
#include "blingfire_sentencebreaker.hpp"
#endif

#ifdef ENABLE_BERT_TOKENIZER
#include "bert_tokenizer.hpp"
#include "basic_tokenizer.hpp"
#include "bert_tokenizer_decoder.hpp"
#endif

#ifdef ENABLE_TRIE_TOKENIZER
#include "trie_tokenizer.hpp"
#endif

FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = []() -> CustomOpArray& {
  static OrtOpLoader op_loader(
#ifdef ENABLE_GPT2_TOKENIZER
      CustomCpuStructV2("GPT2Tokenizer", GPT2Tokenizer),
      CustomCpuStructV2("CLIPTokenizer", CLIPTokenizer),
      CustomCpuStructV2("RobertaTokenizer", RobertaTokenizer),
      CustomCpuStructV2("BpeDecoder", KernelBpeDecoder),
#endif

#ifdef ENABLE_SPM_TOKENIZER
      CustomCpuStruct("SentencepieceTokenizer", KernelSentencepieceTokenizer),
      CustomCpuStruct("SentencepieceDecoder", KernelSentencepieceDecoder),
#endif

#ifdef ENABLE_TRIE_TOKENIZER
      CustomCpuStruct("TrieTokenizer", KernelTrieTokenizer),
      CustomCpuStruct("TrieDetokenizer", KernelTrieDetokenizer),
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
      CustomCpuStruct("WordpieceTokenizer", KernelWordpieceTokenizer),
#endif

#ifdef ENABLE_BERT_TOKENIZER
      CustomCpuStruct("BasicTokenizer", KernelBasicTokenizer),
      CustomCpuStruct("BertTokenizer", KernelBertTokenizer),
      CustomCpuStruct("BertTokenizerDecoder", KernelBertTokenizerDecoder),
      CustomCpuStruct("HfBertTokenizer", KernelHfBertTokenizer),
#endif

#ifdef ENABLE_BLINGFIRE
      CustomCpuStruct("BlingFireSentenceBreaker", KernelBlingFireSentenceBreaker),
#endif
      []() { return nullptr; });

  return op_loader.GetCustomOps();
};

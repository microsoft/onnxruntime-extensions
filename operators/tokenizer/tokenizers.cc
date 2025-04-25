// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"

#ifdef ENABLE_GPT2_TOKENIZER
#include "bpe_kernels.h"
#include "bpe_tokenizer_model.hpp"
#include "bpe_decoder.hpp"
#include "tokenizer_op_impl.hpp"
using namespace ort_extensions;
#endif

#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.h"
#include "sentencepiece_decoder.hpp"
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
#include "wordpiece_tokenizer.h"
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
      CustomCpuStructV2("SpmTokenizer", SpmTokenizer),
      CustomCpuStructV2("HfJsonTokenizer", JsonTokenizerOpKernel),
#endif

#ifdef ENABLE_SPM_TOKENIZER
      CustomCpuStructV2("SentencepieceTokenizer", KernelSentencepieceTokenizer),
      CustomCpuStructV2("SentencepieceDecoder", KernelSentencepieceDecoder),
#endif

#ifdef ENABLE_TRIE_TOKENIZER
      CustomCpuStructV2("TrieTokenizer", KernelTrieTokenizer),
      CustomCpuStructV2("TrieDetokenizer", KernelTrieDetokenizer),
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

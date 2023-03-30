
#include "ocos.h"

#ifdef ENABLE_GPT2_TOKENIZER
#include "gpt2_tokenizer.hpp"
#include "clip_tokenizer.hpp"
#include "roberta_tokenizer.hpp"
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

const std::vector<const OrtCustomOp*>& TokenizerLoader() {
  static OrtOpLoader op_loader(
      []() { return nullptr; }
#ifdef ENABLE_GPT2_TOKENIZER
      ,
      LiteCustomOpStruct("GPT2Tokenizer", KernelBpeTokenizer),
      LiteCustomOpStruct("CLIPTokenizer", KernelClipBpeTokenizer),
      LiteCustomOpStruct("RobertaTokenizer", KernelRobertaBpeTokenizer),
      LiteCustomOpStruct("BpeDecoder", KernelBpeDecoder)
#endif

#ifdef ENABLE_SPM_TOKENIZER
          ,
      LiteCustomOpStruct("SentencepieceTokenizer", KernelSentencepieceTokenizer),
      LiteCustomOpStruct("SentencepieceDecoder", KernelSentencepieceDecoder)
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
          ,
      LiteCustomOpStruct("WordpieceTokenizer", KernelWordpieceTokenizer)
#endif

#ifdef ENABLE_BERT_TOKENIZER
          ,
      LiteCustomOpStruct("BasicTokenizer", KernelBasicTokenizer),
      LiteCustomOpStruct("BertTokenizer", KernelBertTokenizer),
      LiteCustomOpStruct("BertTokenizerDecoder", KernelBertTokenizerDecoder),
      LiteCustomOpStruct("HfBertTokenizer", KernelHfBertTokenizer)
#endif

#ifdef ENABLE_BLINGFIRE
          ,
      LiteCustomOpStruct("BlingFireSentenceBreaker", KernelBlingFireSentenceBreaker)
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = TokenizerLoader;

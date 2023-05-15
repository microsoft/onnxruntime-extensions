
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
      CustomCpuStruct("GPT2Tokenizer", KernelBpeTokenizer),
      CustomCpuStruct("CLIPTokenizer", KernelClipBpeTokenizer),
      CustomCpuStruct("RobertaTokenizer", KernelRobertaBpeTokenizer),
      CustomCpuStruct("BpeDecoder", KernelBpeDecoder)
#endif

#ifdef ENABLE_SPM_TOKENIZER
          ,
      CustomCpuStruct("SentencepieceTokenizer", KernelSentencepieceTokenizer),
      CustomCpuStruct("SentencepieceDecoder", KernelSentencepieceDecoder)
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
          ,
      CustomCpuStruct("WordpieceTokenizer", KernelWordpieceTokenizer)
#endif

#ifdef ENABLE_BERT_TOKENIZER
          ,
      CustomCpuStruct("BasicTokenizer", KernelBasicTokenizer),
      CustomCpuStruct("BertTokenizer", KernelBertTokenizer),
      CustomCpuStruct("BertTokenizerDecoder", KernelBertTokenizerDecoder),
      CustomCpuStruct("HfBertTokenizer", KernelHfBertTokenizer)
#endif

#ifdef ENABLE_BLINGFIRE
          ,
      CustomCpuStruct("BlingFireSentenceBreaker", KernelBlingFireSentenceBreaker)
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = TokenizerLoader;

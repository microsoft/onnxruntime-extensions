
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
      BuildCustomOp(CustomOpRobertaBpeTokenizer),
      BuildCustomOp(CustomOpBpeDecoder)
#endif

#ifdef ENABLE_SPM_TOKENIZER
          ,
      BuildCustomOp(CustomOpSentencepieceTokenizer),
      BuildCustomOp(CustomOpSentencepieceDecoder)
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
          ,
      BuildCustomOp(CustomOpWordpieceTokenizer)
#endif

#ifdef ENABLE_BERT_TOKENIZER
          ,
      BuildCustomOp(CustomOpBasicTokenizer),
      BuildCustomOp(CustomOpBertTokenizer),
      BuildCustomOp(CustomOpBertTokenizerDecoder),
      BuildCustomOp(CustomOpHfBertTokenizer)
#endif

#ifdef ENABLE_BLINGFIRE
          ,
      BuildCustomOp(CustomOpBlingFireSentenceBreaker)
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = TokenizerLoader;

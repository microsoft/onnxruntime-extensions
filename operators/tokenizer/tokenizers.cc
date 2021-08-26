#include "ocos.h"

#ifdef ENABLE_GPT2_TOKENIZER
#include "gpt2_tokenizer.hpp"
#endif 

#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BERT_TOKENIZER
#include "wordpiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BLINGFIRE
#include "blingfire_sentencebreaker.hpp"
#endif


FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = &LoadCustomOpClasses<
#ifdef ENABLE_GPT2_TOKENIZER
    CustomOpBpeTokenizer
#endif
#ifdef ENABLE_SPM_TOKENIZER
    , CustomOpSentencepieceTokenizer
#endif
#ifdef ENABLE_BERT_TOKENIZER
    , CustomOpWordpieceTokenizer
#endif
#ifdef ENABLE_BLINGFIRE
    , CustomOpBlingFireSentenceBreaker
#endif
>;

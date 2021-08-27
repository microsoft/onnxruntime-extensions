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
#if defined ENABLE_GPT2_TOKENIZER
    // comma required only when previous tokenizer is defined,
    // otherwise it will throw build error: expected expression.
    ,
#endif
    CustomOpSentencepieceTokenizer
#endif

#ifdef ENABLE_BERT_TOKENIZER
#if defined ENABLE_GPT2_TOKENIZER || defined ENABLE_SPM_TOKENIZER
    // comma required only when previous tokenizer is defined
    ,
#endif
    CustomOpWordpieceTokenizer
#endif

#ifdef ENABLE_BLINGFIRE
#if defined ENABLE_GPT2_TOKENIZER || defined ENABLE_SPM_TOKENIZER || defined ENABLE_BERT_TOKENIZER
    // comma required only when previous tokenizer is defined
    ,
#endif
    CustomOpBlingFireSentenceBreaker
#endif
>;

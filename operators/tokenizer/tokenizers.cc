
#include "ocos.h"

#ifdef ENABLE_GPT2_TOKENIZER
#include "gpt2_tokenizer.hpp"
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


FxLoadCustomOpFactory LoadCustomOpClasses_Tokenizer = LoadCustomOpClasses<
    CustomOpClassBegin
#ifdef ENABLE_GPT2_TOKENIZER
    , CustomOpBpeTokenizer
#endif

#ifdef ENABLE_SPM_TOKENIZER
    , CustomOpSentencepieceTokenizer
    , CustomOpSentencepieceDecoder
#endif

#ifdef ENABLE_WORDPIECE_TOKENIZER
    , CustomOpWordpieceTokenizer
#endif

#ifdef ENABLE_BERT_TOKENIZER
    , CustomOpBasicTokenizer
    , CustomOpBertTokenizer
    , CustomOpBertTokenizerDecoder
    , CustomOpHfBertTokenizer
#endif

#ifdef ENABLE_BLINGFIRE
    , CustomOpBlingFireSentenceBreaker
#endif
>;

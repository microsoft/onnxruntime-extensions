// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>

#include "string_utils.h"

#include "text/op_equal.hpp"
#include "text/op_segment_sum.hpp"
#include "text/op_ragged_tensor.hpp"
#include "text/string_hash.hpp"
#include "text/string_join.hpp"
#include "text/string_lower.hpp"
#include "text/string_regex_replace.hpp"
#include "text/string_regex_split.hpp"
#include "text/string_split.hpp"
#include "text/string_to_vector.hpp"
#include "text/string_upper.hpp"
#include "text/vector_to_string.hpp"
#include "text/string_length.hpp"
#include "text/string_concat.hpp"


#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BERT_TOKENIZER
#include "wordpiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BLINGFIRE
#include "blingfire_sentencebreaker.hpp"
#endif

#ifdef ENABLE_SPM_TOKENIZER
CustomOpSentencepieceTokenizer c_CustomOpSentencepieceTokenizer;
#endif

#ifdef ENABLE_BERT_TOKENIZER
CustomOpWordpieceTokenizer c_CustomOpWordpieceTokenizer;
#endif

#ifdef ENABLE_TF_STRING
CustomOpSegmentSum c_CustomOpSegmentSum;
CustomOpRaggedTensorToDense c_CustomOpRaggedTensorToDense;
CustomOpRaggedTensorToSparse c_CustomOpRaggedTensorToSparse;
CustomOpStringEqual c_CustomOpStringEqual;
CustomOpStringHash c_CustomOpStringHash;
CustomOpStringHashFast c_CustomOpStringHashFast;
CustomOpStringJoin c_CustomOpStringJoin;
CustomOpStringLower c_CustomOpStringLower;
CustomOpStringRaggedTensorToDense c_CustomOpStringRaggedTensorToDense;
CustomOpStringRegexReplace c_CustomOpStringRegexReplace;
CustomOpStringRegexSplitWithOffsets c_CustomOpStringRegexSplitWithOffsets;
CustomOpStringSplit c_CustomOpStringSplit;
CustomOpStringToVector c_CustomOpStringToVector;
CustomOpStringUpper c_CustomOpStringUpper;
CustomOpVectorToString c_CustomOpVectorToString;
CustomOpStringLength c_CustomOpStringLength;
CustomOpStringConcat c_CustomOpStringConcat;
CustomOpBlingFireSentenceBreaker c_CustomOpTextToSentences;
#endif

OrtCustomOp* operator_lists[] = {
#ifdef ENABLE_SPM_TOKENIZER
    &c_CustomOpSentencepieceTokenizer,
#endif

#ifdef ENABLE_BERT_TOKENIZER
    &c_CustomOpWordpieceTokenizer,
#endif

#ifdef ENABLE_TF_STRING
    &c_CustomOpRaggedTensorToDense,
    &c_CustomOpRaggedTensorToSparse,
    &c_CustomOpSegmentSum,
    &c_CustomOpStringEqual,
    &c_CustomOpStringHash,
    &c_CustomOpStringHashFast,
    &c_CustomOpStringJoin,
    &c_CustomOpStringLower,
    &c_CustomOpStringRaggedTensorToDense,
    &c_CustomOpStringRegexReplace,
    &c_CustomOpStringRegexSplitWithOffsets,
    &c_CustomOpStringSplit,
    &c_CustomOpStringToVector,
    &c_CustomOpStringUpper,
    &c_CustomOpVectorToString,
    &c_CustomOpStringLength,
    &c_CustomOpStringConcat,
    &c_CustomOpTextToSentences,
#endif
    nullptr};

#if ENABLE_MATH
extern FxLoadCustomOpFactory LoadCustomOpClasses_Math;
#endif //ENABLE_MATH

class ExternalCustomOps {
 public:
  ExternalCustomOps() {
  }

  static ExternalCustomOps& instance() {
    static ExternalCustomOps g_instance;
    return g_instance;
  }

  void Add(const OrtCustomOp* c_op) {
    op_array_.push_back(c_op);
  }

  const OrtCustomOp* GetNextOp(size_t& idx) {
    if (idx >= op_array_.size()) {
      return nullptr;
    }

    return op_array_[idx++];
  }

  ExternalCustomOps(ExternalCustomOps const&) = delete;
  void operator=(ExternalCustomOps const&) = delete;

 private:
  std::vector<const OrtCustomOp*> op_array_;
};

extern "C" bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op) {
  ExternalCustomOps::instance().Add(c_op);
  return true;
}



extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);
  std::set<std::string> pyop_nameset;

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    if (auto status = ortApi->CustomOpDomain_Add(domain, c_ops)) {
      return status;
    } else {
      pyop_nameset.emplace(c_ops->GetName(c_ops));
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

  static std::vector<FxLoadCustomOpFactory> c_factories = {
    []() { return const_cast<const OrtCustomOp**>(operator_lists); }
#if defined(ENABLE_MATH)
    ,
    LoadCustomOpClasses_Math
#endif
#if defined(ENABLE_GPT2_TOKENIZER)
    ,
    LoadTokenizerSchemaList
#endif
  };

  for (auto fx : c_factories) {
    auto ops = fx();
    while (*ops != nullptr) {
      if (pyop_nameset.find((*ops)->GetName(*ops)) == pyop_nameset.end()) {
        if (auto status = ortApi->CustomOpDomain_Add(domain, *ops)) {
          return status;
        }
      }
      ++ops;
    }
  }

  size_t idx = 0;
  const OrtCustomOp* e_ops = ExternalCustomOps::instance().GetNextOp(idx);
  while (e_ops != nullptr) {
    if (pyop_nameset.find(e_ops->GetName(e_ops)) == pyop_nameset.end()) {
      if (auto status = ortApi->CustomOpDomain_Add(domain, e_ops)) {
        return status;
      }
      e_ops = ExternalCustomOps::instance().GetNextOp(idx);
    }
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>

#include "kernels/op_equal.hpp"
#include "kernels/op_segment_sum.hpp"
#include "kernels/op_ragged_tensor.hpp"
#include "kernels/string_hash.hpp"
#include "kernels/string_join.hpp"
#include "kernels/string_lower.hpp"
#include "kernels/string_regex_replace.hpp"
#include "kernels/string_regex_split.hpp"
#include "kernels/string_split.hpp"
#include "kernels/string_to_vector.hpp"
#include "kernels/string_upper.hpp"
#include "kernels/negpos.hpp"
#include "kernels/vector_to_string.hpp"
#include "kernels/string_length.hpp"
#include "kernels/string_concat.hpp"
#include "utils/string_utils.h"

#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.hpp"
#endif

#ifdef ENABLE_BERT_TOKENIZER
#include "wordpiece_tokenizer.hpp"
#endif

#ifdef ENABLE_SPM_TOKENIZER
CustomOpSentencepieceTokenizer c_CustomOpSentencepieceTokenizer;
#endif

#ifdef ENABLE_BERT_TOKENIZER
CustomOpWordpieceTokenizer c_CustomOpWordpieceTokenizer;
#endif

#ifdef ENABLE_TF_STRING
CustomOpNegPos c_CustomOpNegPos;
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
#endif

OrtCustomOp* operator_lists[] = {
#ifdef ENABLE_SPM_TOKENIZER
    &c_CustomOpSentencepieceTokenizer,
#endif

#ifdef ENABLE_BERT_TOKENIZER
    &c_CustomOpWordpieceTokenizer,
#endif

#ifdef ENABLE_TF_STRING
    &c_CustomOpNegPos,
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
#endif
    nullptr};

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

extern "C" bool AddExternalCustomOp(const OrtCustomOp* c_op) {
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

  OrtCustomOp** ops = operator_lists;
  while (*ops != nullptr) {
    if (pyop_nameset.find((*ops)->GetName(*ops)) == pyop_nameset.end()) {
      if (auto status = ortApi->CustomOpDomain_Add(domain, *ops)) {
        return status;
      }
    }
    ++ops;
  }

#if defined(ENABLE_GPT2_TOKENIZER)
  const OrtCustomOp** t_ops = LoadTokenizerSchemaList();
  while (*t_ops != nullptr) {
    if (pyop_nameset.find((*t_ops)->GetName(*t_ops)) == pyop_nameset.end()) {
      if (auto status = ortApi->CustomOpDomain_Add(domain, *t_ops)) {
        return status;
      }
    }
    t_ops++;
  }
#endif

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

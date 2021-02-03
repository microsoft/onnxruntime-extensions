// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>

#include "kernels/op_equal.hpp"
#include "kernels/op_segment_sum.hpp"
#include "kernels/op_ragged_tensor.hpp"
#include "kernels/string_hash.hpp"
#include "kernels/string_join.hpp"
#include "kernels/string_regex_replace.hpp"
#include "kernels/string_split.hpp"
#include "kernels/string_upper.hpp"
#include "kernels/negpos.hpp"
#include "utils.h"

#ifdef ENABLE_SPM_TOKENIZER
#include "sentencepiece_tokenizer.hpp"
#endif

CustomOpNegPos c_CustomOpNegPos;
CustomOpSegmentSum c_CustomOpSegmentSum;
CustomOpRaggedTensorToSparse c_CustomOpRaggedTensorToSparse;
#ifdef ENABLE_SPM_TOKENIZER
CustomOpSentencepieceTokenizer c_CustomOpSentencepieceTokenizer;
#endif
CustomOpStringEqual c_CustomOpStringEqual;
CustomOpStringHash c_CustomOpStringHash;
CustomOpStringHashFast c_CustomOpStringHashFast;
CustomOpStringJoin c_CustomOpStringJoin;
CustomOpStringRegexReplace c_CustomOpStringRegexReplace;
CustomOpStringSplit c_CustomOpStringSplit;
CustomOpStringUpper c_CustomOpStringUpper;

OrtCustomOp* operator_lists[] = {
    &c_CustomOpNegPos,
    &c_CustomOpRaggedTensorToSparse,
    &c_CustomOpSegmentSum,
#ifdef ENABLE_SPM_TOKENIZER
    &c_CustomOpSentencepieceTokenizer,
#endif
    &c_CustomOpStringEqual,
    &c_CustomOpStringHash,
    &c_CustomOpStringHashFast,
    &c_CustomOpStringJoin,
    &c_CustomOpStringRegexReplace,
    &c_CustomOpStringSplit,
    &c_CustomOpStringUpper,
    nullptr};


class ExternalCustomOps
{
 public:
  ExternalCustomOps(){
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

    return op_array_[idx ++];
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
  const OrtCustomOp* e_ops =  ExternalCustomOps::instance().GetNextOp(idx);
  while (e_ops != nullptr) {
    if (pyop_nameset.find(e_ops->GetName(e_ops)) == pyop_nameset.end()) {
      if (auto status = ortApi->CustomOpDomain_Add(domain, e_ops)){
        return status;
      }
      e_ops = ExternalCustomOps::instance().GetNextOp(idx);
    }
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

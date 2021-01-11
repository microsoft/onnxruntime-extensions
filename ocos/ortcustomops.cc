// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/op_equal.hpp"
#include "kernels/op_segment_sum.hpp"
#include "kernels/string_hash.hpp"
#include "kernels/string_join.hpp"
#include "kernels/string_regex_replace.hpp"
#include "kernels/string_split.hpp"
#include "kernels/string_upper.hpp"
#include "kernels/test_output.hpp"
#include "utils.h"

CustomOpNegPos c_CustomOpNegPos;
CustomOpSegmentSum c_CustomOpSegmentSum;
CustomOpStringEqual c_CustomOpStringEqual;
CustomOpStringHash c_CustomOpStringHash;
CustomOpStringHashFast c_CustomOpStringHashFast;
CustomOpStringJoin c_CustomOpStringJoin;
CustomOpStringRegexReplace c_CustomOpStringRegexReplace;
CustomOpStringSplit c_CustomOpStringSplit;
CustomOpStringUpper c_CustomOpStringUpper;
CustomOpOne c_CustomOpOne;
CustomOpTwo c_CustomOpTwo;

OrtCustomOp* operator_lists[] = {
    &c_CustomOpNegPos,
    &c_CustomOpSegmentSum,
    &c_CustomOpStringEqual,
    &c_CustomOpStringHash,
    &c_CustomOpStringHashFast,
    &c_CustomOpStringJoin,
    &c_CustomOpStringRegexReplace,
    &c_CustomOpStringSplit,
    &c_CustomOpStringUpper,
    &c_CustomOpOne,
    &c_CustomOpTwo,
    nullptr};

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

  OrtCustomOp** ops = operator_lists;
  while (*ops != nullptr) {
    if (auto status = ortApi->CustomOpDomain_Add(domain, *ops)) {
      return status;
    }
    ++ops;
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    if (auto status = ortApi->CustomOpDomain_Add(domain, c_ops)) {
      return status;
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

#if defined(ENABLE_TOKENIZER)
  auto** t_ops = LoadTokenizerSchemaList();
  while (*t_ops != nullptr) {
    if (auto status = ortApi->CustomOpDomain_Add(domain, *t_ops)){
      return status;
    }
    t_ops++;
  }
#endif

  return ortApi->AddCustomOpDomain(options, domain);
}

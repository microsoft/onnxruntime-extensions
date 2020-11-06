// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "kernels/string_join.hpp"
#include "kernels/string_upper.hpp"
#include "kernels/test_output.hpp"
#include "utils.h"

CustomOpStringUpper c_CustomOpStringUpper;
CustomOpStringJoin c_CustomOpStringJoin;
CustomOpOne c_CustomOpOne;
CustomOpTwo c_CustomOpTwo;
CustomOpNegPos c_CustomOpNegPos;

OrtCustomOp* operator_lists[] = {
    &c_CustomOpStringUpper,
    &c_CustomOpStringJoin,
    &c_CustomOpOne,
    &c_CustomOpTwo,
    &c_CustomOpNegPos,
    nullptr};

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
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
    OrtCustomOp* op_ptr = const_cast<OrtCustomOp*>(c_ops);
    if (auto status = ortApi->CustomOpDomain_Add(domain, op_ptr)) {
      return status;
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

  return ortApi->AddCustomOpDomain(options, domain);
}

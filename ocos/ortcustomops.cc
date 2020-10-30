// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "kernels/kernels.h"
#include "kernels/dummy_kernels.hpp"
#include "kernels/string_kernels.hpp"
#include "utils.h"

#include <vector>
#include <cmath>
#include <algorithm>

OrtCustomOp* get_operator_lists(size_t& n_ops) {
  static OrtCustomOp operator_lists[] = {
      CustomOpStringUpper(),
      CustomOpStringJoin(),
      CustomOpOne(),
      CustomOpTwo(),
      CustomOpNegPos()};
  n_ops = sizeof(operator_lists) / sizeof(operator_lists[0]);
  return operator_lists;
}

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  std::cout << "reg\n";
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);

  if (auto status = ortApi->CreateCustomOpDomain("test.customop", &domain)) {
    std::cout << "domain\n";
    return status;
  } else {
    std::cout << "ops\n";
    size_t n_ops;
    OrtCustomOp* ops = get_operator_lists(n_ops);
    for (size_t i = 0; i < n_ops; ++i) {
      if (auto status = ortApi->CustomOpDomain_Add(domain, &ops[i])) {
        std::cout << "reg " << i << "/" << n_ops << "\n";
        return status;
      }
    }
  }

  domain = nullptr;
  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    OrtCustomOp* op_ptr = const_cast<OrtCustomOp*>(c_ops);
    auto status = ortApi->CustomOpDomain_Add(domain, op_ptr);
    if (status) {
      return status;
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

  return ortApi->AddCustomOpDomain(options, domain);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <set>

#include "onnxruntime_extensions.h"
#include "ocos.h"


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
  OrtStatus* status = nullptr;

#if defined(PYTHON_OP_SUPPORT)
  if (status = RegisterPythonDomainAndOps(options, ortApi)){
    return status;
  }
#endif // PYTHON_OP_SUPPORT

  if (status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) {
    return status;
  }

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    if (status = ortApi->CustomOpDomain_Add(domain, c_ops)) {
      return status;
    } else {
      pyop_nameset.emplace(c_ops->GetName(c_ops));
    }
    ++count;
    c_ops = FetchPyCustomOps(count);
  }
#endif

  static std::vector<FxLoadCustomOpFactory> c_factories = {
    LoadCustomOpClasses<CustomOpClassBegin>
#if defined(ENABLE_TF_STRING)
    , LoadCustomOpClasses_Text
#endif // ENABLE_TF_STRING
#if defined(ENABLE_MATH)
    , LoadCustomOpClasses_Math
#endif
#if defined(ENABLE_TOKENIZER)
    , LoadCustomOpClasses_Tokenizer
#endif
  };

  for (auto fx : c_factories) {
    auto ops = fx();
    while (*ops != nullptr) {
      if (pyop_nameset.find((*ops)->GetName(*ops)) == pyop_nameset.end()) {
        if (status = ortApi->CustomOpDomain_Add(domain, *ops)) {
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
      if (status = ortApi->CustomOpDomain_Add(domain, e_ops)) {
        return status;
      }
      e_ops = ExternalCustomOps::instance().GetNextOp(idx);
    }
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

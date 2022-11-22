// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <set>

#include "onnxruntime_extensions.h"
#include "ocos.h"
#include "vision/vision.hpp"

struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

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

extern "C" ORTX_EXPORT OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtCustomOpDomain* domain = nullptr;
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION);
  std::set<std::string> pyop_nameset;
  OrtStatus* status = nullptr;

#if defined(PYTHON_OP_SUPPORT)
  if (status = RegisterPythonDomainAndOps(options, ortApi); status) {
    return status;
  }
#endif  // PYTHON_OP_SUPPORT

  if (status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain); status) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

#if defined(PYTHON_OP_SUPPORT)
  size_t count = 0;
  const OrtCustomOp* c_ops = FetchPyCustomOps(count);
  while (c_ops != nullptr) {
    if (status = ortApi->CustomOpDomain_Add(domain, c_ops); status) {
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
    ,
    LoadCustomOpClasses_Text
#endif  // ENABLE_TF_STRING
#if defined(ENABLE_MATH)
    ,
    LoadCustomOpClasses_Math
#endif
#if defined(ENABLE_TOKENIZER)
    ,
    LoadCustomOpClasses_Tokenizer
#endif
#if defined(ENABLE_CV2)
    ,
    LoadCustomOpClasses_CV2
#endif
  };

  for (const auto& fx : c_factories) {
    const auto& ops = fx();
    for (const OrtCustomOp* op : ops) {
      if (pyop_nameset.find(op->GetName(op)) == pyop_nameset.end()) {
        if (status = ortApi->CustomOpDomain_Add(domain, op); status) {
          return status;
        }
      }
    }
  }

  size_t idx = 0;
  const OrtCustomOp* e_ops = ExternalCustomOps::instance().GetNextOp(idx);
  while (e_ops != nullptr) {
    if (pyop_nameset.find(e_ops->GetName(e_ops)) == pyop_nameset.end()) {
      if (status = ortApi->CustomOpDomain_Add(domain, e_ops); status) {
        return status;
      }
      e_ops = ExternalCustomOps::instance().GetNextOp(idx);
    }
  }

  if (status = ortApi->AddCustomOpDomain(options, domain); status) {
    return status;
  }

  // Create domain for ops using the new domain name.
  if (status = ortApi->CreateCustomOpDomain(c_ComMsExtOpDomain, &domain); status) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  static std::vector<FxLoadCustomOpFactory> new_domain_factories = {
    LoadCustomOpClasses<CustomOpClassBegin>
#if defined(ENABLE_VISION)
    ,
    LoadCustomOpClasses_PPP_Vision
#endif
  };

  for (const auto& fx : new_domain_factories) {
    const auto& ops = fx();
    for (const OrtCustomOp* op : ops) {
      if (status = ortApi->CustomOpDomain_Add(domain, op); status) {
        return status;
      }
    }
  }

  return ortApi->AddCustomOpDomain(options, domain);
}

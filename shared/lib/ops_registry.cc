// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <mutex>
#include <set>
#include <cstdlib>  // for std::atoi
#include <string>

#include "onnxruntime_extensions.h"
#include "ocos.h"

using namespace OrtW;

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

static int GetOrtVersion(const OrtApiBase* api_base = nullptr) noexcept {
  // the version will be cached after the first call on RegisterCustomOps
  static int ort_version = MIN_ORT_VERSION_SUPPORTED;  // the default version is 1.11.0

  if (api_base != nullptr) {
    std::string str_version = api_base->GetVersionString();

    std::size_t first_dot = str_version.find('.');
    if (first_dot != std::string::npos) {
      std::size_t second_dot = str_version.find('.', first_dot + 1);
      // If there is no second dot and the string has more than one character after the first dot, set second_dot to the string length
      if (second_dot == std::string::npos && first_dot + 1 < str_version.length()) {
        second_dot = str_version.length();
      }

      if (second_dot != std::string::npos) {
        std::string str_minor_version = str_version.substr(first_dot + 1, second_dot - first_dot - 1);
        int ver = std::atoi(str_minor_version.c_str());
        // Only change ort_version if conversion is successful (non-zero value)
        if (ver != 0) {
          ort_version = ver;
        }
      }
    }
  }

  return ort_version;
}

extern "C" bool ORT_API_CALL AddExternalCustomOp(const OrtCustomOp* c_op) {
  OCOS_API_IMPL_BEGIN
  ExternalCustomOps::instance().Add(c_op);
  OCOS_API_IMPL_END
  return true;
}

extern "C" int ORT_API_CALL GetActiveOrtAPIVersion() {
  int ver = 0;
  OCOS_API_IMPL_BEGIN
  ver = GetOrtVersion();
  OCOS_API_IMPL_END
  return ver;
}

// The main entrance of the extension library.
extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) {
  OrtStatus* status = nullptr;
  OCOS_API_IMPL_BEGIN

  // the following will initiate some global objects which
  //  means any other function invocatoin prior to these calls to trigger undefined behavior.
  auto ver = GetOrtVersion(api);
  const OrtApi* ortApi = api->GetApi(ver);
  API::instance(ortApi);

  OrtCustomOpDomain* domain = nullptr;
  std::set<std::string> pyop_nameset;

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
#if defined(ENABLE_TF_STRING)
      LoadCustomOpClasses_Text,
#endif  // ENABLE_TF_STRING
#if defined(ENABLE_MATH)
      LoadCustomOpClasses_Math,
#endif
#if defined(ENABLE_TOKENIZER)
      LoadCustomOpClasses_Tokenizer,
#endif
#if defined(ENABLE_CV2)
      LoadCustomOpClasses_CV2,
#endif
#if defined(ENABLE_DR_LIBS)
      LoadCustomOpClasses_Audio,
#endif
#if defined(USE_CUDA)
      LoadCustomOpClasses_Contrib,
#endif
      LoadCustomOpClasses<>,
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

  //
  // New custom ops should use the com.microsoft.extensions domain.
  //

  // Create domain for ops using the new domain name.
  if (status = ortApi->CreateCustomOpDomain(c_ComMsExtOpDomain, &domain); status) {
    return status;
  }

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  static std::vector<FxLoadCustomOpFactory> new_domain_factories = {
#if defined(ENABLE_VISION)
      LoadCustomOpClasses_Vision,
#endif
#if defined(ENABLE_TOKENIZER)
      LoadCustomOpClasses_Tokenizer,
#endif
#if defined(ENABLE_AZURE)
      LoadCustomOpClasses_Azure,
#endif
      LoadCustomOpClasses<>};

  for (const auto& fx : new_domain_factories) {
    const auto& ops = fx();
    for (const OrtCustomOp* op : ops) {
      if (status = ortApi->CustomOpDomain_Add(domain, op); status) {
        return status;
      }
    }
  }

  status = ortApi->AddCustomOpDomain(options, domain);

  OCOS_API_IMPL_END

  return status;
}

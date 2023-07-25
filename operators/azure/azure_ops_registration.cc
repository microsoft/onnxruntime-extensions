// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "azure_basic_invokers.hpp"
#include "azure_triton_invoker.hpp"

using namespace ort_extensions;

const std::vector<const OrtCustomOp*>& AzureInvokerLoader() {
  static OrtOpLoader op_loader(CustomAzureStruct("AzureAudioToText", AzureAudioToText),
                               CustomAzureStruct("AzureTextToText", AzureTextToText),
                               CustomAzureStruct("AzureTritonInvoker", AzureTritonInvoker)
#ifdef TEST_AZURE_INVOKERS_AS_CPU_OP
                                   ,
                               CustomCpuStruct("AzureAudioToText", AzureAudioToText),
                               CustomCpuStruct("AzureTextToText", AzureTextToText),
                               CustomCpuStruct("AzureTritonInvoker", AzureTritonInvoker)
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;

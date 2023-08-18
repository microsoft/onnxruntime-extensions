// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "azure_invokers.hpp"
#include "openai_invokers.hpp"

#ifdef AZURE_INVOKERS_ENABLE_TRITON
#include "azure_triton_invoker.hpp"
#endif

using namespace ort_extensions;

#ifdef AZURE_OP_AS_CPU_OP
#define AZURE_OP CustomCpuStruct
#else
#define AZURE_OP CustomAzureStruct
#endif

const std::vector<const OrtCustomOp*>& AzureInvokerLoader() {
  static OrtOpLoader op_loader(AZURE_OP("AzureAudioToText", AzureAudioToTextInvoker),
                               AZURE_OP("AzureTextToText", AzureTextToTextInvoker),
                               AZURE_OP("OpenAIAudioToText", OpenAIAudioToTextInvoker)
#ifdef AZURE_INVOKERS_ENABLE_TRITON
                                   ,
                               AZURE_OP("AzureTritonInvoker", AzureTritonInvoker)
#endif
  );

  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;

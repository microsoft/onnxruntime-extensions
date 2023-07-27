// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "azure_invokers.hpp"
#include "openai_invokers.hpp"

#ifdef AZURE_INVOKERS_ENABLE_TRITON
#include "azure_triton_invoker.hpp"
#endif 

using namespace ort_extensions;

const std::vector<const OrtCustomOp*>& AzureInvokerLoader() {
  static OrtOpLoader op_loader(CustomAzureStruct("AzureAudioToText", AzureAudioToTextInvoker),
                               CustomAzureStruct("AzureTextToText", AzureTextToTextInvoker),
                               CustomAzureStruct("OpenAIAudioToText", OpenAIAudioToText)
#ifdef AZURE_INVOKERS_ENABLE_TRITON
                               ,
                               CustomAzureStruct("AzureTritonInvoker", AzureTritonInvoker)
#endif
#ifdef TEST_AZURE_INVOKERS_AS_CPU_OP
                               ,
                               CustomCpuStruct("AzureAudioToText", AzureAudioToTextInvoker),
                               CustomCpuStruct("AzureTextToText", AzureTextToTextInvoker),
                               CustomCpuStruct("OpenAIAudioToText", OpenAIAudioToText)
#ifdef AZURE_INVOKERS_ENABLE_TRITON
                               ,
                               CustomCpuStruct("AzureTritonInvoker", AzureTritonInvoker)
#endif
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "ocos.h"
#include "azure_invokers.hpp"
#include "azure_triton_invoker.hpp"
#include "openai_invokers.hpp"

using namespace ort_extensions;

const std::vector<const OrtCustomOp*>& AzureInvokerLoader() {
  static OrtOpLoader op_loader(CustomAzureStruct("AzureAudioToText", AzureAudioToText),
                               CustomAzureStruct("AzureTextToText", AzureTextToText),
                               CustomAzureStruct("AzureTritonInvoker", AzureTritonInvoker),
                               CustomAzureStruct("OpenAIAudioToText", OpenAIAudioToText)
#ifdef TEST_AZURE_INVOKERS_AS_CPU_OP
                                   ,
                               CustomCpuStruct("AzureAudioToText", AzureAudioToText),
                               CustomCpuStruct("AzureTextToText", AzureTextToText),
                               CustomCpuStruct("AzureTritonInvoker", AzureTritonInvoker),
                               CustomCpuStruct("OpenAIAudioToText", OpenAIAudioToText)
#endif
  );
  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;

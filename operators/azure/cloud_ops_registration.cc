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
                               CustomCpuStruct("AzureAudioToText", AzureAudioToTextInvoker),
                               CustomAzureStruct("AzureTextToText", AzureTextToTextInvoker),
                               CustomCpuStruct("AzureTextToText", AzureTextToTextInvoker),
                               CustomAzureStruct("OpenAIAudioToText", OpenAIAudioToTextInvoker),
                               CustomCpuStruct("OpenAIAudioToText", OpenAIAudioToTextInvoker)
#ifdef AZURE_INVOKERS_ENABLE_TRITON
                                   ,
                               CustomAzureStruct("AzureTritonInvoker", AzureTritonInvoker),
                               CustomCpuStruct("AzureTritonInvoker", AzureTritonInvoker)
#endif
  );

  return op_loader.GetCustomOps();
}

FxLoadCustomOpFactory LoadCustomOpClasses_Azure = AzureInvokerLoader;

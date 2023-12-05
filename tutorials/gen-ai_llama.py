# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: This script cannot be run independently and is simply a tutorial. Please build the onnxruntime-genai project
# located here: https://github.com/microsoft/onnxruntime-genai/tree/main
# and replace src/pybind/llama.py with this script for correct functionality.

import onnxruntime_genai as og
import numpy as np
from transformers import LlamaTokenizer
from onnxruntime_extensions import OrtPyFunction, gen_processing_models

device_type = og.DeviceType.CPU
#device_type = og.DeviceType.CUDA

# Generate input tokens from the text prompt
tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

ort_tok = OrtPyFunction.from_model(gen_processing_models(
    tokenizer,
    pre_kwargs={"WITH_DEFAULT_INPUTS": True})[0])

print("Loading model...")
#model=og.Llama_Model("../../test_models/llama2-7b-fp32-cpu/Llama-2-7b-hf_decoder_merged_model_fp32_opt.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-fp16-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_fp16.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-int4-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_int4.onnx", device_type)
#model=og.Model("../../test_models/llama2-7b-chat-int4-gpu", device_type)
model=og.Model("../../test_models/llama2-7b-fp32-cpu/", device_type)
print("Model loaded")

# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = ort_tok([text])[0]
    print("ORT Extensions Tokenized Input IDs: " + str(input_tokens))

    params=og.SearchParams(model)
    params.max_length = 128
    params.input_ids = input_tokens

    search=og.GreedySearch(params, model.DeviceType)
    state=model.CreateState(search.GetSequenceLengths(), params)

    print("Output:")

    print(text, end='', flush=True)
    while not search.IsDone():
        search.SetLogits(state.Run(search.GetSequenceLength(), search.GetNextTokens()))

        # search.Apply_MinLength(1)
        # search.Apply_RepetitionPenalty(1.0)

        search.SampleTopP(0.7, 0.6)

        # Print each token as we compute it, we have to do some work to get newlines & spaces to appear properly:
        word=tokenizer.convert_ids_to_tokens([search.GetNextTokens().GetArray()[0]])[0]
        if word=='<0x0A>':
          word = '\n'
        if word.startswith('▁'):
          word = ' ' + word[1:]
        print(word, end='', flush=True)

    print()
    print()

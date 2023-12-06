# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# NOTE: This script cannot be run independently and is simply a tutorial. Please build the onnxruntime-genai project
# located here: https://github.com/microsoft/onnxruntime-genai/tree/main
# and replace src/pybind/llama.py with this script for correct functionality.

import onnxruntime_genai as og
import numpy as np
import onnxruntime_extensions as _ortx

device_type = og.DeviceType.CPU
#device_type = og.DeviceType.CUDA

# ort-tokenizer can be converted with following model
# from transformers import LlamaTokenizer
# tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
# _ortx.gen_processing_models(
#    tokenizer).save("../test_models/llama2-tokenizer.json")


print("Loading model...")
#model=og.Llama_Model("../../test_models/llama2-7b-fp32-cpu/Llama-2-7b-hf_decoder_merged_model_fp32_opt.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-fp16-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_fp16.onnx", device_type)
#model=og.Llama_Model("../../test_models/llama2-7b-int4-gpu/rank_0_Llama-2-7b-hf_decoder_merged_model_int4.onnx", device_type)
#model=og.Model("../../test_models/llama2-7b-chat-int4-gpu", device_type)
model=og.Model("../../test_models/llama2-7b-fp32-cpu/", device_type)
print("Model loaded")
tokenizer = _ortx.LoadTokenizer("../test_models/llama2-tokenizer.json")

# Keep asking for input prompts in an loop
while True:
    text = input("Input:")
    input_tokens = tokenizer([text])[0]

    params=og.SearchParams(model)
    params.max_length = 128
    params.input_ids = input_tokens

    search=og.GreedySearch(params, model.DeviceType)
    state=model.CreateState(search.GetSequenceLengths(), params)

    print("Output:")

    print(text, end='', flush=True)
    while not search.IsDone():
        search.SetLogits(state.Run(search.GetSequenceLength(), search.GetNextTokens()))

        search.Apply_MinLength(1)
        search.Apply_RepetitionPenalty(1.0)

        search.SampleTopP(0.7, 0.6)

    decoded_text = tokenizer.Detokenizer(search.GetSequence(0))
    print(decoded_text)

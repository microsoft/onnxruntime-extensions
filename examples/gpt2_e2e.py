import os
import numpy
from transformers import AutoConfig
from onnxruntime_customops import ONNXTraceSession, EagerOp, mytorch as torch


cache_dir = os.path.join("../..", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

device = 'cpu'
model_name_or_path = 'gpt2'
gpt2_core_model_path = './gpt2.onnx'
gpt2_full_model_path = './gpt2_full.onnx'

config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)

# check if the gpt2 core model is already exported.
if not os.path.exists(gpt2_core_model_path):
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel
    model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    Gpt2Helper.export_onnx(model, device, gpt2_core_model_path)

core_model = EagerOp.from_model(gpt2_core_model_path)

gpt2_tokenizer = EagerOp.from_customop('GPT2Tokenizer', merges='merges.txt', vacab='vacab.txt')
gpt2_decoder = EagerOp.from_customop('VectorToString', map='map.txt')

input_text = ['best hotel in bay area', 'here is an example of gpt2 model']
num_tokens_to_produce = 10

with ONNXTraceSession(inputs=[input_text]) as ox:
    num_attention_heads = config.n_head
    hidden_size = config.n_embd
    num_layer = config.n_layer
    vocab_size = config.vocab_size
    eos_token_id = config.eos_token_id

    encodings_dict = gpt2_tokenizer(input_text, padding=True, padding_side='left')

    input_ids = torch.tensor(encodings_dict['input_ids'], dtype=torch.int64)
    attention_mask = torch.tensor(encodings_dict['attention_mask'], dtype=torch.float32)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    position_ids.masked_fill_(position_ids < 0, 0)

    # Empty Past State for generating first word
    empty_past = []
    batch_size = input_ids.size(0)
    sequence_length = input_ids.size(1)
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    for i in range(num_layer):
        empty_past.append(torch.empty(past_shape).type(torch.float32).to(device))      

    has_eos = torch.zeros(batch_size, dtype=torch.bool)

    all_token_ids = input_ids.clone()

    all_eos = 0
    for step in torch.onnx_loop(num_tokens_to_produce, all_eos):
        outputs = core_model(input_ids, position_ids, attention_mask, past)

        next_token_logits = outputs[0][:, -1, :]
        next_tokens = torch.argmax(next_token_logits, dim=-1)

        has_eos = has_eos | (next_tokens == eos_token_id)
        tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
        all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

        # Update input_ids, attention_mask, position_ids and past
        input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)    
        position_ids = (position_ids[:, -1] + 1).reshape(batch_size, 1)
        attention_mask = torch.cat([attention_mask, torch.ones([batch_size, 1]).type_as(attention_mask)], 1).to(device)    

        past = []
        for i in range(num_layer):
            past_i = torch.from_numpy(outputs[i + 1]) if \
                isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()
            past.append(past_i.to(device))

        all_eos = torch.all(has_eos)

    output_text = gpt2_decoder(all_token_ids)
    ox.set_outputs([output_text])
    ox.save_as_onnx(gpt2_full_model_path)


full_model = EagerOp.from_model(gpt2_full_model_path)
outputs = full_model(input_text)
for n_, tx_ in enumerate(outputs):
    print("{}: {}\n\t{}".format(n_, input_text, tx_))

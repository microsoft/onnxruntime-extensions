import os
import numpy
from transformers import AutoConfig
from onnxruntime_customops import mytorch as torch
from onnxruntime_customops.utils import trace_for_onnx, op_from_model, build_customop_model


device = 'cpu'
model_name_or_path = 'gpt2'
gpt2_core_model_path = './gpt2.onnx'
gpt2_encoder_model_path = './gpt2_tok.onnx'
gpt2_decoder_model_path = './gpt2_dec.onnx'
gpt2_full_model_path = './gpt2_full.onnx'


# input_text = ['best hotel in bay area', 'here is an example of gpt2 model']
input_text = ['best hotel in bay area']
num_tokens_to_produce = 10


def get_cache_directory():
    cache_dir = os.path.join("../..", "cache_models")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    return cache_dir


def convert_models():
    from transformers import GPT2Tokenizer  # noqa
    from onnxruntime.transformers.gpt2_helper import Gpt2Helper, MyGPT2LMHeadModel  # noqa
    cache_dir = get_cache_directory()

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    build_customop_model('GPT2Tokenizer', gpt2_encoder_model_path, model=tokenizer)
    build_customop_model('VectorToString', gpt2_decoder_model_path, decoder=tokenizer.decoder)

    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    # remove the dependency from onnxruntime-tools.
    model = MyGPT2LMHeadModel.from_pretrained(model_name_or_path, config=config, cache_dir=cache_dir)
    Gpt2Helper.export_onnx(model, device, gpt2_core_model_path)


def inference_and_dump_full_model(inputs):
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=get_cache_directory())
    core_model = op_from_model(gpt2_core_model_path)
    gpt2_tokenizer = op_from_model(gpt2_encoder_model_path)

    with trace_for_onnx(inputs, names=gpt2_tokenizer.input_names) as tc_sess:

        num_attention_heads = config.n_head
        hidden_size = config.n_embd
        num_layer = config.n_layer
        eos_token_id = config.eos_token_id

        input_ids, attention_mask = gpt2_tokenizer(*tc_sess.get_inputs(), padding=True, padding_side='left')

        position_ids = (attention_mask.long().cumsum(-1) - 1)
        # position_ids.masked_fill_(position_ids < 0, 0)

        # Empty Past State for generating first word
        batch_size = input_ids.size()[0]
        past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
        empty_past = [torch.empty(*past_shape).type(torch.float32).to(device)] * num_layer

        has_eos = torch.zeros(batch_size, dtype=torch.bool)

        all_token_ids = input_ids.clone()
        all_eos = torch.tensor(False, dtype=torch.bool)
        past = empty_past
        for step in torch.onnx_loop(num_tokens_to_produce, all_eos, past):
            outputs = core_model(input_ids, position_ids, attention_mask.type(torch.float32), *past)

            next_token_logits = outputs[0][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            has_eos = has_eos | (next_tokens == eos_token_id)
            tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)
            all_token_ids = torch.cat([all_token_ids, tokens_to_add.unsqueeze(-1)], dim=-1)

            # FIXME: not support the loop yet.
            break

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

        gpt2_decoder = torch.op_from_model(gpt2_decoder_model_path)
        output_text = gpt2_decoder(all_token_ids.squeeze(0))

        tc_sess.save_as_onnx(gpt2_full_model_path, output_text)
        return output_text


# 1. Check if the gpt2 models is already exported, otherwise, they are converted.
if not os.path.exists(gpt2_core_model_path):
    convert_models()

# 2. Run the inference with the pre and post process, trace the computation graph and build the all-in-one ONNX model
output_ms = inference_and_dump_full_model(input_text)

# 3. Inference on the all-in-one model
full_model = torch.op_from_model(gpt2_full_model_path)
outputs = full_model(input_text)

# 4. Test the result
for n_, tx_ in enumerate(outputs):
    if output_ms[n_] != outputs[n_]:
        import warnings  # noqa
        warnings.warn("{}: The all-in-one model output is not the same\t{}".format(n_, output_ms[n_]))
    print("{}: {}\n\t{}".format(n_, input_text, tx_))

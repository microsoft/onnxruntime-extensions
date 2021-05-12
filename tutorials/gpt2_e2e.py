import os
import numpy
from transformers import AutoConfig
from onnxruntime_extensions import mytorch as torch, eager_op
from onnxruntime_extensions.utils import trace_for_onnx, op_from_model, build_customop_model


device = 'cpu'
model_name_or_path = 'gpt2'
gpt2_core_model_path = './gpt2.onnx'
gpt2_encoder_model_path = './gpt2_tok.onnx'
gpt2_decoder_model_path = './gpt2_dec.onnx'
gpt2_full_model_path = './gpt2_full.onnx'


input_text = ['best hotel in bay area', 'here is an example of gpt2 model']
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

    with trace_for_onnx(inputs, num_tokens_to_produce, names=gpt2_tokenizer.input_names) as tc_sess:

        num_attention_heads = config.n_head
        hidden_size = config.n_embd
        num_layer = config.n_layer
        eos_token_id = config.eos_token_id

        inputs, num_tokens = tc_sess.get_inputs()
        input_ids, attention_mask = gpt2_tokenizer(inputs, padding=True, padding_side='left')
        attention_mask = attention_mask.type(torch.float)
        position_ids = (attention_mask.long().cumsum(-1) - 1)
        # position_ids.masked_fill_(position_ids < 0, 0)

        # Empty Past State for generating first word
        batch_size = input_ids.size()[0]
        past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
        empty_past = []
        for _ in range(num_layer):
            empty_past.append(torch.empty(*past_shape).type(torch.float32).to(device))

        has_eos = torch.zeros(batch_size, dtype=torch.bool)

        all_eos = torch.tensor(False, dtype=torch.bool)
        past = empty_past
        cfg = torch.control_flow()
        for states in cfg.loop(num_tokens, ~all_eos, has_eos,
                               input_ids, position_ids, attention_mask.type(torch.float), *past):
            _, has_eos, input_ids, position_ids, attention_mask, *past = states
        # for _ in [1]:
            outputs = core_model(input_ids, position_ids, attention_mask, *past)
            next_token_logits = outputs[0][:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            has_eos = has_eos | (next_tokens == eos_token_id)
            tokens_to_add = next_tokens.masked_fill(has_eos, eos_token_id)

            # Update input_ids, attention_mask, position_ids and past
            batch_size = 2
            input_ids = tokens_to_add.clone().detach().reshape([batch_size, 1]).to(device)
            position_ids = (position_ids[:, -1] + 1).reshape([batch_size, 1])
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1).type(torch.float)], 1).to(device)

            all_eos = torch.all(has_eos)

            past = []
            for i in range(num_layer):
                past_i = torch.from_numpy(outputs[i + 1]) if \
                    isinstance(outputs[i + 1], numpy.ndarray) else outputs[i + 1].clone().detach()
                past.append(past_i.to(device))

            cfg.flow_output(~all_eos, has_eos, input_ids, position_ids, attention_mask, *past, tokens_to_add)

        *_, all_token_ids = cfg.finalize()
        gpt2_decoder = torch.op_from_model(gpt2_decoder_model_path)
        # text_out = gpt2_decoder(tokens_to_add.unsqueeze(-1))
        text_out = gpt2_decoder(all_token_ids.transpose(0, 1)[0].unsqueeze(-1))
        tc_sess.save_as_onnx(gpt2_full_model_path, text_out)
        return text_out


# 1. Check if the gpt2 models is already exported, otherwise, they are converted.
if not os.path.exists(gpt2_core_model_path) or \
        not os.path.exists(gpt2_decoder_model_path) or \
        not os.path.exists(gpt2_encoder_model_path):
    convert_models()


# 2. Run the inference with the pre and post process, trace the computation graph and build the all-in-one ONNX model
output_ms = inference_and_dump_full_model(input_text)

# 3. Inference on the all-in-one model
full_model = eager_op.EagerOp.from_model(gpt2_full_model_path)
output_text = full_model(input_text, num_tokens_to_produce)

# 4. Test the result
if not numpy.array_equal(output_ms.numpy(), output_text):
    import warnings  # noqa
    warnings.warn("{}: The all-in-one model output is not the same\t{}".format(output_ms.numpy(), output_text))

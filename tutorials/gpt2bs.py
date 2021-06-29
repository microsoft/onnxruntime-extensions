import os
import numpy
import argparse
from transformers import AutoConfig


model_name_or_path = "gpt2"
device = "cpu"
default_beam_width = 4
default_batch_size = 1
onnx_model_path = "gpt2_one_step_search.onnx"
gpt2_full_model_path = "gpt2_full.onnx"

# Create a cache directory to store pretrained model.
cache_dir = os.path.expanduser('~/.cache/huggingface/')
if not os.path.exists(cache_dir):
    cache_dir = os.path.join(".", "cache_models")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


def _extract_endict(tokenizer_endict):
    _1, _2 = [tokenizer_endict.get(ky_) for ky_ in ('input_ids', 'attention_mask')]
    return (_1, _2.astype(numpy.float32))


def get_tokenizer(model_name_or_path, enable_tokenizer, cache_dir):
    from transformers import GPT2Tokenizer  # noqa
    from onnxruntime_extensions.onnxprocess import build_customop_model, pyfunc_from_model

    tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)

    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    if enable_tokenizer:
        gpt2_encoder_model_path = './gpt2_tok.onnx'
        build_customop_model('GPT2Tokenizer', gpt2_encoder_model_path, model=tokenizer)
        return tokenizer, pyfunc_from_model(gpt2_encoder_model_path)
    else:
        return tokenizer, None


def convert_gpt2():
    import onnxruntime
    from distutils.version import StrictVersion
    if StrictVersion(onnxruntime.__version__) < StrictVersion('1.8'):
        raise RuntimeError('Full GPT-2 model is only available on onxruntime 1.8 and higher version.')
    from onnxruntime.transformers.gpt2_beamsearch_helper import Gpt2BeamSearchHelper, GPT2LMHeadModel_BeamSearchStep
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    model = GPT2LMHeadModel_BeamSearchStep.from_pretrained(
        model_name_or_path, config=config, batch_size=default_batch_size, beam_size=default_beam_width, cache_dir=cache_dir)
    model.eval().to(device)
    Gpt2BeamSearchHelper.export_onnx(model, device, onnx_model_path)


def inference_and_dump_full_model(tokenizer, func_tokenizer, input_text, num_tokens_to_produce = 30):
    from onnxruntime_extensions.onnxprocess import trace_for_onnx, pyfunc_from_model

    func_one_step = pyfunc_from_model(onnx_model_path)
    config = AutoConfig.from_pretrained(model_name_or_path, cache_dir=cache_dir)
    num_attention_heads = config.n_head
    hidden_size = config.n_embd
    num_layer = config.n_layer
    if func_tokenizer is None:
        input_ids, attention_mask = _extract_endict(tokenizer(input_text, padding=True, return_tensors='np'))
        with trace_for_onnx(input_ids, attention_mask,
                            num_tokens_to_produce, names=["input_ids", "attention_mask", "out_token_num"], target_opset=12) as tc_sess:
            input_ids, attention_mask, num_tokens = tc_sess.get_inputs()
            _beam_search(tokenizer, func_one_step, num_attention_heads, hidden_size, num_layer, tc_sess, num_tokens, input_ids, attention_mask)
    else:
        with trace_for_onnx(input_text, num_tokens_to_produce, names=func_tokenizer.input_names, target_opset=12) as tc_sess:
            inputs, num_tokens = tc_sess.get_inputs()
            input_ids, attention_mask = func_tokenizer(inputs, padding=True)
            _beam_search(tokenizer, func_one_step, num_attention_heads, hidden_size, num_layer, tc_sess, num_tokens, input_ids, attention_mask)


def _beam_search(tokenizer, func_one_step, num_attention_heads, hidden_size, num_layer, tc_sess, num_tokens, input_ids, attention_mask):
    from onnxruntime_extensions.onnxprocess import torch_wrapper as torch

    if attention_mask.dtype is not torch.float32:
        attention_mask = attention_mask.type(torch.float)
    position_ids = (attention_mask.long().cumsum(-1) - 1)
    batch_size = default_batch_size
    past_shape = [2, batch_size, num_attention_heads, 0, hidden_size // num_attention_heads]
    empty_past = []
    for _ in range(num_layer):
        empty_past.append(torch.empty(*past_shape).type(torch.float32).to(device))

    beam_select_idx = torch.zeros([1, batch_size]).long()
    input_log_probs = torch.zeros([batch_size, 1])
    input_unfinished_sents = torch.ones([batch_size, 1], dtype=torch.bool)
    prev_step_scores = torch.zeros([batch_size, 1])
    beam_size = default_beam_width
    prev_step_results = input_ids.clone().detach().to(device)
    
    cfg = torch.control_flow()
    for states in cfg.loop(num_tokens, torch.tensor(True), input_ids, position_ids,
                               attention_mask, beam_select_idx, input_log_probs,
                               input_unfinished_sents, prev_step_results, prev_step_scores, *empty_past):
        step = states[0]
        states[1].symbolic_shape = ['batch_size', 'seq_len']
        states[2].symbolic_shape = ['batch_size', 'seq_len']
        states[3].symbolic_shape = ['batch_size', 'all_seq_len']
        states[4].symbolic_shape = [1, 'batch_size']

            # prev_step_results
        states[7].symbolic_shape = ['batch_size', 'total_seq_len']

        for st_ in states[-num_layer:]:
            st_.symbolic_shape = [2, 'batch_size', num_attention_heads, 'past_seq_len', hidden_size // num_attention_heads]

        prev_attention_mask = states[3]
        outputs = func_one_step(*states[1:])
        last_state = outputs[0].clone().detach().cpu()
        input_ids = last_state.reshape([batch_size * beam_size, -1]).to(device)

        input_unfinished_sents_id = -3
        prev_step_results = outputs[-2].clone().detach().to(device)
            # position_ids = (torch.tensor([context_length + step - 1
            #                                     ]).unsqueeze(0).repeat(batch_size * beam_size, 1).to(device))
        position_ids = torch.zeros([batch_size * beam_size, 1], dtype=torch.int64) + attention_mask.size()[-1]
        factor = (~step.type(torch.bool)).type(torch.int64)
        prev_attention_mask = prev_attention_mask.repeat(factor * (batch_size * beam_size - 1) + 1, 1).to(device)
        attention_mask = torch.cat(
                [
                    prev_attention_mask,
                    torch.ones([batch_size * beam_size, 1], dtype=torch.float),
                ],
                1,
            ).to(device)

        beam_select_idx = outputs[input_unfinished_sents_id - 2].clone().detach().to(device)
        input_log_probs = outputs[input_unfinished_sents_id - 1].clone().detach().to(device)
        input_unfinished_sents = outputs[input_unfinished_sents_id].clone().detach().to(device)
        prev_step_scores = outputs[-1].clone().detach().to(device)

        past = []
        for i in range(num_layer):
            past_i =  outputs[i + 1].clone().detach()
            past.append(past_i.to(device))


        any_unfinished = input_unfinished_sents.any()
        input_ids.symbolic_shape = ['total_batch_size', 'seq_len']
        position_ids.symbolic_shape = ['total_batch_size', 'seq_len']
        attention_mask.symbolic_shape = ['total_batch_size', 'all_seq_len']
        prev_step_results.symbolic_shape = ['total_batch_size', 'step_seq_len']
        for st_ in past:
            st_.symbolic_shape = [2, 'total_batch_size', num_attention_heads, 'all_seq_len', hidden_size // num_attention_heads]
        cfg.flow_output(any_unfinished, input_ids,
                            position_ids, attention_mask, beam_select_idx,
                            input_log_probs, input_unfinished_sents, prev_step_results, prev_step_scores, *past)

    result_id = 6
    all_token_ids = cfg.finalize()[result_id]
    tc_sess.save_as_onnx(gpt2_full_model_path, all_token_ids)

    print(tokenizer.decode(all_token_ids.t[0], skip_special_tokens=True))


def verify_bsfull_model(input_text, tokenizer, enable_tokenizer):
    import time
    from onnxruntime_extensions import PyOrtFunction
    gpt2_all = PyOrtFunction.from_model(gpt2_full_model_path)
    gpt2_all._ensure_ort_session()
    if enable_tokenizer:
        start_time = time.perf_counter()
        outputs = gpt2_all(input_text, 30)
    else:
        input_ids, attention_mask = _extract_endict(tokenizer(input_text, padding=True, return_tensors='np'))
        start_time = time.perf_counter()
        outputs = gpt2_all(input_ids, attention_mask, 30)
    print("total time: {}".format(time.perf_counter() - start_time))
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))


def main(enable_tokenizer):
    tokenizer, func_tokenizer = get_tokenizer(model_name_or_path, enable_tokenizer, cache_dir)
    input_text = ['best hotel in bay area.']
    if not os.path.exists(onnx_model_path):
        convert_gpt2()
    inference_and_dump_full_model(tokenizer, func_tokenizer, input_text)
    verify_bsfull_model(input_text, tokenizer, enable_tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable-tokenizer", help="No tokenizer operator for the full model",
                        action="store_true")
    parser.add_argument("--output", '-o', help="The output file name")
    args = parser.parse_args()
    if args.output is not None:
        gpt2_full_model_path = args.output
    main(not args.disable_tokenizer)

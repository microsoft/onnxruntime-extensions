import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto, TensorProto
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort

import math
import os
import platform
import random
import torch
from einops import rearrange, repeat
from onnxruntime import InferenceSession, OrtValue, SessionOptions

torch.manual_seed(0)
class Formats:
    BSNH = 0
    BNSH = 1


class Config:
    batch_size = 0
    sequence_length = 0
    kv_sequence_length = 0
    past_sequence_length = 0
    num_heads = 0
    kv_num_heads = 0
    head_size = 0

    def __init__(self, b, s, s2, sp, n, n2, h):
        self.batch_size = b
        self.sequence_length = s
        self.kv_sequence_length = s2
        self.past_sequence_length = sp
        self.num_heads = n
        self.kv_num_heads = n2
        self.head_size = h

def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = seqlen_k if key_padding_mask is None else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    sq = seqlen_q if query_padding_mask is None else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )

def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)

def create_group_query_attention_graph_past(
    config,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    local_window_size=-1,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
):
    past_kv_seqlen = config.kv_sequence_length
    present_kv_seqlen = (
        config.kv_sequence_length if share_buffer else config.kv_sequence_length + config.sequence_length
    )
    #pdb.set_trace()
    nodes = [
        helper.make_node(
            "GroupQueryAttention",
            [
                "query",
                "key" if not packed else "",
                "value" if not packed else "",
                "past_key",
                "past_value",
                "seqlens_k",
                "total_sequence_length",
#                "cos_cache" if rotary else "",
#                "sin_cache" if rotary else "",
            ],
            ["output", "present_key", "present_value"],
            "GroupQueryAttention_0",
            num_heads=config.num_heads,
            kv_num_heads=config.kv_num_heads,
            local_window_size=local_window_size,
            do_rotary=rotary,
            rotary_interleaved=rotary_interleaved,
            # is_past_bsnh=1 if past_kv_format == Formats.BSNH else 0,
            # kv_share_buffer=1 if share_buffer else 0,
            domain="ai.onnx.contrib",
        ),
    ]

    graph_input = [
        helper.make_tensor_value_info(
            "query",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                config.sequence_length,
                (config.num_heads * config.head_size)
                if not packed
                else (config.num_heads * config.head_size + 2 * config.kv_num_heads * config.head_size),
            ],
        ),
        helper.make_tensor_value_info(
            "past_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "past_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                past_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else past_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "seqlens_k",
            TensorProto.INT32,
            [config.batch_size],
        ),
        helper.make_tensor_value_info(
            "total_sequence_length",
            TensorProto.INT32,
            [1],
        ),
    ]
    if not packed:
        graph_input += [
            helper.make_tensor_value_info(
                "key",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
            helper.make_tensor_value_info(
                "value",
                TensorProto.FLOAT16,
                [
                    config.batch_size,
                    config.sequence_length,
                    config.kv_num_heads * config.head_size,
                ],
            ),
        ]
    if rotary:
        graph_input += [
            helper.make_tensor_value_info(
                "cos_cache",
                TensorProto.FLOAT16,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
            helper.make_tensor_value_info(
                "sin_cache",
                TensorProto.FLOAT16,
                [
                    config.kv_sequence_length + (0 if share_buffer else config.sequence_length),
                    (math.floor(config.head_size / 16) * 16) // 2,
                ],
            ),
        ]

    graph_output = [
        helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT16,
            [config.batch_size, config.sequence_length, config.num_heads * config.head_size],
        ),
        helper.make_tensor_value_info(
            "present_key",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
        helper.make_tensor_value_info(
            "present_value",
            TensorProto.FLOAT16,
            [
                config.batch_size,
                present_kv_seqlen if past_kv_format == Formats.BSNH else config.kv_num_heads,
                config.kv_num_heads if past_kv_format == Formats.BSNH else present_kv_seqlen,
                config.head_size,
            ],
        ),
    ]

    graph = helper.make_graph(
        nodes,
        "GroupQueryAttention_Graph",
        graph_input,
        graph_output,
    )

    model = make_onnx_model(graph)
    return model

def gqa_past_func(
    q,
    k,
    v,
    config,
    new_k,
    new_v,
    cos=None,
    sin=None,
    seqlens_k=None,
    past_kv_format=Formats.BSNH,
    share_buffer=True,
    window_size=-1,
    rotary_interleaved=False,
):
    onnx_model = create_group_query_attention_graph_past(
        config,
        past_kv_format,
        share_buffer,
        local_window_size=window_size,
        rotary=cos is not None,
        rotary_interleaved=rotary_interleaved,
        packed=new_k is None,
    )
    q = torch.reshape(q, (config.batch_size, config.sequence_length, -1))
    past_k = k.clone()
    past_v = v.clone()
    if new_k is not None:
        new_k = torch.reshape(new_k, (config.batch_size, config.sequence_length, -1))
        new_v = torch.reshape(new_v, (config.batch_size, config.sequence_length, -1))
    if share_buffer:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": OrtValue.ortvalue_from_numpy(past_k.detach().cpu().numpy(), "cuda", 0),
            "past_value": OrtValue.ortvalue_from_numpy(past_v.detach().cpu().numpy(), "cuda", 0),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(np.int32),
            "total_sequence_length": torch.tensor([config.kv_sequence_length], dtype=torch.int32)
            .detach()
            .cpu()
            .numpy(),
        }
        sess_options = SessionOptions()
        sess_options.register_custom_ops_library(_get_library_path())
        ort_session = InferenceSession(onnx_model.SerializeToString(), sess_options, providers=["CUDAExecutionProvider"])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_input(
            "past_key", "cuda", 0, np.float16, ort_inputs["past_key"].shape(), ort_inputs["past_key"].data_ptr()
        )
        io_binding.bind_input(
            "past_value",
            "cuda",
            0,
            np.float16,
            ort_inputs["past_value"].shape(),
            ort_inputs["past_value"].data_ptr(),
        )
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_ortvalue_output("present_key", ort_inputs["past_key"])
        io_binding.bind_ortvalue_output("present_value", ort_inputs["past_value"])
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = np.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v
    else:
        ort_inputs = {
            "query": q.detach().cpu().numpy(),
            "past_key": past_k.detach().cpu().numpy(),
            "past_value": past_v.detach().cpu().numpy(),
            "seqlens_k": seqlens_k.detach().cpu().numpy().astype(np.int32),
            "total_sequence_length": torch.tensor(
                [config.kv_sequence_length + config.sequence_length], dtype=torch.int32
            )
            .detach()
            .cpu()
            .numpy(),
        }
        sess_options = SessionOptions()
        sess_options.register_custom_ops_library(_get_library_path())
        ort_session = InferenceSession(onnx_model.SerializeToString(), sess_options, providers=["CUDAExecutionProvider"])
        io_binding = ort_session.io_binding()
        if new_k is not None:
            ort_inputs["key"] = new_k.detach().cpu().numpy()
            ort_inputs["value"] = new_v.detach().cpu().numpy()
            io_binding.bind_cpu_input("key", ort_inputs["key"])
            io_binding.bind_cpu_input("value", ort_inputs["value"])
        if cos is not None:
            ort_inputs["cos_cache"] = cos.detach().cpu().numpy()
            ort_inputs["sin_cache"] = sin.detach().cpu().numpy()
            io_binding.bind_cpu_input("cos_cache", ort_inputs["cos_cache"])
            io_binding.bind_cpu_input("sin_cache", ort_inputs["sin_cache"])
        io_binding.bind_cpu_input("query", ort_inputs["query"])
        io_binding.bind_cpu_input("past_key", ort_inputs["past_key"])
        io_binding.bind_cpu_input("past_value", ort_inputs["past_value"])
        io_binding.bind_cpu_input("seqlens_k", ort_inputs["seqlens_k"])
        io_binding.bind_cpu_input("total_sequence_length", ort_inputs["total_sequence_length"])
        io_binding.bind_output("output")
        io_binding.bind_output("present_key")
        io_binding.bind_output("present_value")
        ort_session.run_with_iobinding(io_binding)
        ort_output, present_k, present_v = io_binding.copy_outputs_to_cpu()
        ort_output = np.array(ort_output)
        output = torch.tensor(ort_output)
        return output, present_k, present_v

def parity_check_gqa_past_no_buff(
    config,
    causal=False,
    local=False,
    past_format=Formats.BSNH,
    rotary=False,
    rotary_interleaved=False,
    packed=False,
    rtol=1e-3,
    atol=1e-3,
):
    torch.manual_seed(69)
    q = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    k = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    v = torch.randn(
        config.batch_size,
        config.kv_sequence_length if past_format == Formats.BSNH else config.kv_num_heads,
        config.kv_num_heads if past_format == Formats.BSNH else config.kv_sequence_length,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_k = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )
    new_v = torch.randn(
        config.batch_size,
        config.sequence_length,
        config.kv_num_heads,
        config.head_size,
        device="cuda",
        dtype=torch.float16,
        requires_grad=False,
    )

    window_size = (-1, -1)
    left_window_size = -1
    if local:
        left_window_size = random.randint(0, config.kv_sequence_length)
        window_size = (left_window_size, 0)
    elif causal:
        left_window_size = -1
        window_size = (-1, 0)

    # Pytorch to compare
    k_cache_ref = k.clone()
    v_cache_ref = v.clone()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)
    k_cache_ref = torch.cat((k_cache_ref, new_k), 1)
    v_cache_ref = torch.cat((v_cache_ref, new_v), 1)
    # cache_seqlens = torch.tensor([config.past_sequence_length], device="cuda").repeat(config.batch_size)
    cache_seqlens = torch.randint(
        0,
        config.kv_sequence_length,
        (config.batch_size,),
        dtype=torch.int32,
        device="cuda",
    )
    cache_seqlens[random.randint(0, config.batch_size - 1)] = config.kv_sequence_length

    cos, sin = None, None
    q_ro, k_ro = q, new_k

    arange = rearrange(torch.arange(config.kv_sequence_length + config.sequence_length, device="cuda"), "s -> 1 s")
    cache_seqlens_expanded = rearrange(cache_seqlens, "b -> b 1")
    update_mask = torch.logical_and(
        cache_seqlens_expanded <= arange, arange < cache_seqlens_expanded + config.sequence_length
    )
    k_cache_ref[update_mask] = rearrange(k_ro, "b s ... -> (b s) ...")
    v_cache_ref[update_mask] = rearrange(new_v, "b s ... -> (b s) ...")
    k_cache_rep = repeat(k_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    v_cache_rep = repeat(v_cache_ref, "b s h d -> b s (h g) d", g=config.num_heads // config.kv_num_heads)
    key_padding_mask = arange < cache_seqlens_expanded + config.sequence_length
    out_ref, _ = attention_ref(
        q_ro, k_cache_rep, v_cache_rep, None, key_padding_mask, 0.0, None, causal=True, window_size=window_size
    )
    out_ref = out_ref.detach().cpu().numpy()
    if past_format == Formats.BNSH:
        k_cache_ref = k_cache_ref.transpose(1, 2)
        v_cache_ref = v_cache_ref.transpose(1, 2)

    # Flash function
    if packed:
        packed_qkv = torch.concatenate([q, new_k, new_v], dim=2)
        out, present_k, present_v = gqa_past_func(
            packed_qkv,
            k,
            v,
            config,
            None,
            None,
            cos,
            sin,
            cache_seqlens,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
        )
    else:
        out, present_k, present_v = gqa_past_func(
            q,
            k,
            v,
            config,
            new_k,
            new_v,
            cos,
            sin,
            cache_seqlens,
            past_format,
            False,
            window_size=left_window_size,
            rotary_interleaved=rotary_interleaved,
        )
    out = torch.squeeze(out, 0)
    out = torch.reshape(out, (config.batch_size, config.sequence_length, config.num_heads, config.head_size))
    out = out.detach().cpu().numpy()
    # Compare results
    print(
        "NO buff",
        " packed:",
        packed,
        " causal:",
        causal,
        " local:",
        local,
        " rotary:",
        rotary,
        " rotary_interleaved:",
        rotary_interleaved,
        "past kv format:",
        "BSNH" if past_format == Formats.BSNH else "BNSH",
        " B:",
        config.batch_size,
        " S:",
        config.sequence_length,
        " kv S:",
        config.kv_sequence_length,
        " N:",
        config.num_heads,
        " kv N:",
        config.kv_num_heads,
        " h:",
        config.head_size,
        " Mean Error:",
        np.mean(np.abs(out - out_ref)),
        np.allclose(
            out,
            out_ref,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        ),
    )

class TestCudaOps(unittest.TestCase):
    @staticmethod
    def _create_negpos_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node('Identity', ['x'], ['identity1']),
            helper.make_node(
                'NegPos', ['identity1'], ['neg', 'pos'],
                domain=domain)
        ]

        input0 = helper.make_tensor_value_info(
            'x', onnx_proto.TensorProto.FLOAT, [None, None])
        output1 = helper.make_tensor_value_info(
            'neg', onnx_proto.TensorProto.FLOAT, [None, None])
        output2 = helper.make_tensor_value_info(
            'pos', onnx_proto.TensorProto.FLOAT, [None, None])

        graph = helper.make_graph(nodes, 'test0', [input0], [output1, output2])
        model = make_onnx_model(graph)
        return model

    def test_cuda_negpos(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_negpos_test_model()
        self.assertIn('op_type: "NegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        x = np.array([[0., 1., 1.5], [7., 8., -5.5]]).astype(np.float32)
        neg, pos = sess.run(None, {'x': x})
        diff = x - (neg + pos)
        assert_almost_equal(diff, np.zeros(diff.shape))

    @staticmethod
    def _create_fastgelu_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                'FastGelu', ['x', 'bias'], ['y'],
                domain=domain)
        ]

        input0 = helper.make_tensor_value_info(
            'x', onnx_proto.TensorProto.FLOAT, [])
        input1 = helper.make_tensor_value_info(
            'bias', onnx_proto.TensorProto.FLOAT, [])
        output0 = helper.make_tensor_value_info(
            'y', onnx_proto.TensorProto.FLOAT, [])

        graph = helper.make_graph(nodes, 'test1', [input0, input1], [output0])
        model = make_onnx_model(graph)
        return model

    @staticmethod
    def _create_fastgelu_test_model_f16(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                'FastGelu', ['x', 'bias'], ['y'],
                domain=domain)
        ]

        input0 = helper.make_tensor_value_info(
            'x', onnx_proto.TensorProto.FLOAT16, [])
        input1 = helper.make_tensor_value_info(
            'bias', onnx_proto.TensorProto.FLOAT16, [])
        output0 = helper.make_tensor_value_info(
            'y', onnx_proto.TensorProto.FLOAT16, [])

        graph = helper.make_graph(nodes, 'test1', [input0, input1], [output0])
        model = make_onnx_model(graph)
        return model

    def test_cuda_fastgelu(self):
        eps = _ort.get_available_providers()
        if 'CUDAExecutionProvider' in eps:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model = self._create_fastgelu_test_model()
            self.assertIn('op_type: "FastGelu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                         so,
                                         providers=['CUDAExecutionProvider'])
            x = np.array([0., 1., 2., 3., 4., 5.]).astype(np.float32)
            bias = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float32)
            expected_y = np.array([0., 0.9505811, 2.1696784, 3.298689, 4.399991, 5.5]).astype(np.float32)
            y = sess.run(None, {'x': x, 'bias':bias})[0]
            assert_almost_equal(y, expected_y)
        else:
            print ('CUDAExecutionProvider not available, test_cuda_fastgelu skipped.')

    def test_cuda_fastgelu_f16(self):
        eps = _ort.get_available_providers()
        if 'CUDAExecutionProvider' in eps:
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model = self._create_fastgelu_test_model_f16()
            self.assertIn('op_type: "FastGelu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                         so,
                                         providers=['CUDAExecutionProvider'])
            x = np.array([0., 1., 2., 3., 4., 5.]).astype(np.float16)
            bias = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]).astype(np.float16)
            expected_y = np.array([0., 0.95, 2.17, 3.299, 4.4, 5.5]).astype(np.float16)
            y = sess.run(None, {'x': x, 'bias':bias})[0]
            assert_almost_equal(y, expected_y)
        else:
            print ('CUDAExecutionProvider not available, test_cuda_fastgelu_f16 skipped.')

    @staticmethod
    def _create_GroupQueryAttention_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                'GroupQueryAttention', 
                #['query', 'key', 'value', 'past_key', 'past_value', 'seqlens_k', 'total_seqlen', 'cos_cache', 'sin_cache'], 
                ['query', 'key', 'value', 'past_key', 'past_value', 'seqlens_k', 'total_seqlen'], 
                ['attn_out', 'present_key', 'present_value'],
                #domain=domain, num_heads=32, kv_num_heads=32, scale=0.0, local_window_size=-1, do_rotary=0, rotary_interleaved=0)
                domain=domain, num_heads=32, kv_num_heads=32)
        ]

        query = helper.make_tensor_value_info(
            'query', onnx_proto.TensorProto.FLOAT16, [5,1,512])
        key = helper.make_tensor_value_info(
            'key', onnx_proto.TensorProto.FLOAT16, [5,1,512])
        value = helper.make_tensor_value_info(
            'value', onnx_proto.TensorProto.FLOAT16, [5,1,512])
        past_key = helper.make_tensor_value_info(
            'past_key', onnx_proto.TensorProto.FLOAT16, [5,32,128,16])
        past_value = helper.make_tensor_value_info(
            'past_value', onnx_proto.TensorProto.FLOAT16, [5,32,128,16])
        seqlens_k = helper.make_tensor_value_info(
            'seqlens_k', onnx_proto.TensorProto.INT32, [5])
        total_seqlen = helper.make_tensor_value_info(
            'total_seqlen', onnx_proto.TensorProto.INT32, [1])
#        cos_cache = helper.make_tensor_value_info(
#            'cos_cache', onnx_proto.TensorProto.FLOAT, [])
#        sin_cache = helper.make_tensor_value_info(
#            'sin_cache', onnx_proto.TensorProto.FLOAT, [])
        attn_out = helper.make_tensor_value_info(
            'attn_out', onnx_proto.TensorProto.FLOAT16, [5,1,512])
        present_key = helper.make_tensor_value_info(
            'present_key', onnx_proto.TensorProto.FLOAT16, [5,32,129,16])
        present_value = helper.make_tensor_value_info(
            'present_value', onnx_proto.TensorProto.FLOAT16, [5,32,129,16])

        graph = helper.make_graph(nodes, 'testgqa', 
                    #[query, key, value, past_key, past_value, seqlens_k, total_seqlen, cos_cache, sin_cache], 
                    [query, key, value, past_key, past_value, seqlens_k, total_seqlen], 
                    [attn_out, present_key, present_value])
        model = make_onnx_model(graph)
        return model

    def test_cuda_GroupQueryAttention(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_GroupQueryAttention_test_model()
        #self.assertIn('op_type: "NegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        query = np.random.randn(5,1,512).astype(np.float16)
        key = np.random.randn(5,1,512).astype(np.float16)
        value = np.random.randn(5,1,512).astype(np.float16)
        past_key = np.random.randn(5,32,128,16).astype(np.float16)
        past_value = np.random.randn(5,32,128,16).astype(np.float16)
        seqlens_k = np.array([128, 87, 0, 22, 125]).astype(np.int32)
        total_seqlen = np.array([129]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'past_key':past_key, 'past_value':past_value, 'seqlens_k':seqlens_k, 'total_seqlen':total_seqlen})

    def test_cuda_GroupQueryAttention_iobinding(self):
        random.seed(69)
        for b in [5]:
            for s, s2 in [(1,128)]:
                for n, n2 in [(32, 32)]:
                    for h in [16]:
                        for past_kv_format in [Formats.BNSH]:
                            sp = random.randint(1, s2 - s) if s2 - s > 0 else 0
                            config = Config(b, s, s2, sp, n, n2, h)
                            parity_check_gqa_past_no_buff(
                                config,
                                past_format=past_kv_format,
                                rtol=1e-3,
                                atol=1e-3,
                            )

    @staticmethod
    def _create_GroupQueryAttention_test_model_validate_PA(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                'GroupQueryAttention', 
                ['query', 'key', 'value', 'seqlens_k', 'total_seqlen'], 
                ['attn_out'],
                domain=domain, num_heads=32, kv_num_heads=32)
        ]

        query = helper.make_tensor_value_info(
            'query', onnx_proto.TensorProto.FLOAT16, [5,34,512])
        key = helper.make_tensor_value_info(
            'key', onnx_proto.TensorProto.FLOAT16, [5,34,512])
        value = helper.make_tensor_value_info(
            'value', onnx_proto.TensorProto.FLOAT16, [5,34,512])
#        past_key = helper.make_tensor_value_info(
#            'past_key', onnx_proto.TensorProto.FLOAT16, [5,32,128,16])
#        past_value = helper.make_tensor_value_info(
#            'past_value', onnx_proto.TensorProto.FLOAT16, [5,32,128,16])
        seqlens_k = helper.make_tensor_value_info(
            'seqlens_k', onnx_proto.TensorProto.INT32, [5])
        total_seqlen = helper.make_tensor_value_info(
            'total_seqlen', onnx_proto.TensorProto.INT32, [1])
        attn_out = helper.make_tensor_value_info(
            'attn_out', onnx_proto.TensorProto.FLOAT16, [5,34,512])

        graph = helper.make_graph(nodes, 'testgqa', 
                    [query, key, value, seqlens_k, total_seqlen], 
                    [attn_out])
        model = make_onnx_model(graph)
        return model

    def test_cuda_GroupQueryAttention_validate_PagedAttention(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_GroupQueryAttention_test_model_validate_PA()
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        query = np.load('query.npy')
        key = np.load('key.npy')
        value = np.load('value.npy')
        query_batch = np.random.randn(5, 34, 512).astype(np.float16)
        key_batch = np.random.randn(5, 34, 512).astype(np.float16)
        value_batch = np.random.randn(5, 34, 512).astype(np.float16)
#        query_batch[0, 0:5] = query[0:5]    # TODO(leca): Need padding for the rest?
#        query_batch[1, 0:12] = query[5:17]
#        query_batch[2, 0:16] = query[17:33]
#        query_batch[3, 0:20] = query[33:53]
#        query_batch[4, 0:34] = query[53:87]
#        key_batch[0, 0:5] = key[0:5]
#        key_batch[1, 0:12] = key[5:17]
#        key_batch[2, 0:16] = key[17:33]
#        key_batch[3, 0:20] = key[33:53]
#        key_batch[4, 0:34] = key[53:87]
#        value_batch[0, 0:5] = value[0:5]
#        value_batch[1, 0:12] = value[5:17]
#        value_batch[2, 0:16] = value[17:33]
#        value_batch[3, 0:20] = value[33:53]
#        value_batch[4, 0:34] = value[53:87]
        seqlens_k = np.array([5, 12, 16, 20, 34]).astype(np.int32)
        total_seqlen = np.array([87]).astype(np.int32)
        y = sess.run(None, {'query':query_batch, 'key':key_batch, 'value':value_batch, 'seqlens_k':seqlens_k, 'total_seqlen':total_seqlen})
        print('y=')
        print(y)


if __name__ == "__main__":
    unittest.main()

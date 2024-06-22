import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort
import torch
from einops import rearrange, repeat
import math
import pdb

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
    def _create_pagedattention_test_model(batch_size, total_seqlen, hidden_size, slot_cnt_per_block, block_cnt_per_layer, block_cnt_needed_by_longest_seq, num_heads=32, num_kv_heads=32, head_size=16, scale=0.0, domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node('PagedAttention',  
                ['query', 'key', 'value', 'key_cache', 'value_cache', 'block_tables', 'slot_mappings', 'context_lens', 'is_prompt'], 
                ['attn_out'], 
                domain=domain, num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=head_size, scale=scale)
        ]
        query = helper.make_tensor_value_info(
            'query', onnx_proto.TensorProto.FLOAT16, [None, hidden_size])
        key = helper.make_tensor_value_info(
            'key', onnx_proto.TensorProto.FLOAT16, [None, hidden_size])
        value = helper.make_tensor_value_info(
            'value', onnx_proto.TensorProto.FLOAT16, [None, hidden_size])
        key_cache = helper.make_tensor_value_info(
            'key_cache', onnx_proto.TensorProto.FLOAT16, [block_cnt_per_layer, hidden_size * slot_cnt_per_block])
        value_cache = helper.make_tensor_value_info(
            'value_cache', onnx_proto.TensorProto.FLOAT16, [block_cnt_per_layer, hidden_size * slot_cnt_per_block])
        block_tables = helper.make_tensor_value_info(
            'block_tables', onnx_proto.TensorProto.INT32, [batch_size, block_cnt_needed_by_longest_seq])
        slot_mappings = helper.make_tensor_value_info(
            'slot_mappings', onnx_proto.TensorProto.INT32, [None])
        context_lens = helper.make_tensor_value_info(
            'context_lens', onnx_proto.TensorProto.INT32, [batch_size])
        is_prompt = helper.make_tensor_value_info(
            'is_prompt', onnx_proto.TensorProto.INT32, [1])
        attn_out = helper.make_tensor_value_info(
            'attn_out', onnx_proto.TensorProto.FLOAT16, [None, hidden_size])
        graph = helper.make_graph(nodes, 'test_paged_attention', 
                    [query, key, value, key_cache, value_cache, block_tables, slot_mappings, context_lens, is_prompt], 
                    [attn_out])
        model = make_onnx_model(graph)
        return model
        
    def test_cuda_paged_attention(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model(5, 87, 512, 16, 32, 3)
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        #query = np.random.randn(87,512).astype(np.float16) # 87 is the token num of all the sequences (5+12+16+20+34)
        #key = np.random.randn(87,512).astype(np.float16)
        #value = np.random.randn(87,512).astype(np.float16)
        query = np.load('query.npy')
        key = np.load('key.npy')
        value = np.load('value.npy')
        key_cache = np.zeros([32,8192]).astype(np.float16)
        value_cache = np.zeros([32,8192]).astype(np.float16)
        block_tables = np.array([[0,-1,-1],[1,-1,-1],[2,-1,-1],[3,4,-1],[5,6,7]]).astype(np.int32)
        slot1 = np.arange(0, 5, dtype=np.int32)
        slot2 = np.arange(16, 28, dtype=np.int32)
        slot3 = np.arange(32, 68, dtype=np.int32)
        slot4 = np.arange(80, 114, dtype=np.int32)
        slot_mappings = np.concatenate((slot1, slot2, slot3, slot4))
        context_lens = np.array([5, 12, 16, 20, 34]).astype(np.int32)
        is_prompt = np.array([1]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'key_cache':key_cache, 'value_cache':value_cache, 'block_tables':block_tables, 'slot_mappings':slot_mappings, 'context_lens':context_lens, 'is_prompt':is_prompt})
        print('Y=')
        print(y)

    def test_cuda_paged_attention2(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model(3, 381, 512, 16, 32, 8)
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        #query = np.random.randn(381,512).astype(np.float16) # 381 is the token num of all the sequences (127, 127, 127)
        #key = np.random.randn(381,512).astype(np.float16)
        #value = np.random.randn(381,512).astype(np.float16)
        query = np.load('query_381x512_float16.npy')
        key = np.load('key_381x512_float16.npy')
        value = np.load('value_381x512_float16.npy')
        key_cache = np.zeros([32,8192]).astype(np.float16)
        value_cache = np.zeros([32,8192]).astype(np.float16)
        block_tables = np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23]]).astype(np.int32) # each sequence occupies 8 blocks (127/16)
        slot1 = np.arange(0, 127, dtype=np.int32)
        slot2 = np.arange(128, 255, dtype=np.int32)
        slot3 = np.arange(256, 383, dtype=np.int32)
        slot_mappings = np.concatenate((slot1, slot2, slot3))
        context_lens = np.array([127, 127, 127]).astype(np.int32)
        is_prompt = np.array([1]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'key_cache':key_cache, 'value_cache':value_cache, 'block_tables':block_tables, 'slot_mappings':slot_mappings, 'context_lens':context_lens, 'is_prompt':is_prompt})
        #pdb.set_trace()
        print('Y=')
        print(y)
        q_pt = torch.from_numpy(query.reshape(3, 127, 32, 16))
        k_pt = torch.from_numpy(key.reshape(3, 127, 32, 16))
        v_pt = torch.from_numpy(value.reshape(3, 127, 32, 16))
        out, attention = attention_ref(q_pt, k_pt, v_pt, causal=True, window_size=[-1, 0])
        y_np = np.array(y).reshape(381, 512)
        out_np = out.reshape(381, 512).numpy()
        assert np.allclose(y_np, out_np, rtol=1e-3, atol=1e-3, equal_nan=True)

    def test_cuda_paged_attention3(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model(3, 381, 512, 16, 32, 8)
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])

        query = np.random.randn(381,512).astype(np.float16) # 381 is the token num of all the sequences (127, 127, 127)
        key = np.random.randn(381,512).astype(np.float16)
        value = np.random.randn(381,512).astype(np.float16)
        key_cache = np.zeros([32,8192]).astype(np.float16)
        value_cache = np.zeros([32,8192]).astype(np.float16)
        block_tables = np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23]]).astype(np.int32) # each sequence occupies 8 blocks (127/16)
        slot1 = np.arange(0, 127, dtype=np.int32)
        slot2 = np.arange(128, 255, dtype=np.int32)
        slot3 = np.arange(256, 383, dtype=np.int32)
        slot_mappings = np.concatenate((slot1, slot2, slot3))
        context_lens = np.array([127, 127, 127]).astype(np.int32)
        is_prompt = np.array([1]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'key_cache':key_cache, 'value_cache':value_cache, 'block_tables':block_tables, 'slot_mappings':slot_mappings, 'context_lens':context_lens, 'is_prompt':is_prompt})
        q_pt = torch.from_numpy(query.reshape(3, 127, 32, 16))
        k_pt = torch.from_numpy(key.reshape(3, 127, 32, 16))
        v_pt = torch.from_numpy(value.reshape(3, 127, 32, 16))
        out, attention = attention_ref(q_pt, k_pt, v_pt, causal=True, window_size=[-1, 0])
        y_np = np.array(y).reshape(381, 512)
        out_np = out.reshape(381, 512).numpy()
        assert np.allclose(y_np, out_np, rtol=1e-3, atol=1e-3, equal_nan=True)

    def test_cuda_paged_attention_prompt_decoding(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model(3, 381, 512, 16, 32, 8)
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
    
        query = np.random.randn(381,512).astype(np.float16) # 381 is the token num of all the sequences (127, 127, 127)
        key = np.random.randn(381,512).astype(np.float16)
        value = np.random.randn(381,512).astype(np.float16)
        key_cache = np.zeros([32,8192]).astype(np.float16)
        value_cache = np.zeros([32,8192]).astype(np.float16)
        block_tables = np.array([[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15],[16,17,18,19,20,21,22,23]]).astype(np.int32) # each sequence occupies 8 blocks (127/16)
        slot1 = np.arange(0, 127, dtype=np.int32)
        slot2 = np.arange(128, 255, dtype=np.int32)
        slot3 = np.arange(256, 383, dtype=np.int32)
        slot_mappings = np.concatenate((slot1, slot2, slot3))
        context_lens = np.array([127, 127, 127]).astype(np.int32)
        is_prompt = np.array([1]).astype(np.int32)
    
        key_cache_ort = _ort.OrtValue.ortvalue_from_numpy(key_cache, "cuda")
        value_cache_ort = _ort.OrtValue.ortvalue_from_numpy(value_cache, "cuda")
        block_tables_ort = _ort.OrtValue.ortvalue_from_numpy(block_tables, "cuda")
        slot_mappings_ort = _ort.OrtValue.ortvalue_from_numpy(slot_mappings, "cuda")
        context_lens_ort = _ort.OrtValue.ortvalue_from_numpy(context_lens)
        is_prompt_ort = _ort.OrtValue.ortvalue_from_numpy(is_prompt)
    
        # prompt case
        io_binding = sess.io_binding()
        io_binding.bind_cpu_input("query", query)
        io_binding.bind_cpu_input("key", key)
        io_binding.bind_cpu_input("value", value)
        io_binding.bind_ortvalue_input("key_cache", key_cache_ort)
        io_binding.bind_ortvalue_input("value_cache", value_cache_ort)
        io_binding.bind_ortvalue_input("block_tables", block_tables_ort)
        io_binding.bind_ortvalue_input("slot_mappings", slot_mappings_ort)
        io_binding.bind_ortvalue_input("context_lens", context_lens_ort)
        io_binding.bind_ortvalue_input("is_prompt", is_prompt_ort)
        io_binding.bind_output("attn_out")
        sess.run_with_iobinding(io_binding)
    
        # decoding case
        query2 = np.random.randn(3, 512).astype(np.float16)
        key2 = np.random.randn(3, 512).astype(np.float16)
        value2 = np.random.randn(3, 512).astype(np.float16)
        slot = np.array([127, 255, 383]).astype(np.int32)
        io_binding.bind_cpu_input("query", query2)
        io_binding.bind_cpu_input("key", key2)
        io_binding.bind_cpu_input("value", value2)
        io_binding.bind_cpu_input("slot_mappings", slot)
        context_lens_ort.update_inplace(np.array([1,1,1]).astype(np.int32))
        is_prompt_ort.update_inplace(np.array([0]).astype(np.int32))
        sess.run_with_iobinding(io_binding)
    
    def _generate_block_kvcache(seqlen_k, paged_kv_block_size, batch_size, nheads_k, d, device, dtype):
        pdb.set_trace()
        num_blocks = math.ceil(seqlen_k / paged_kv_block_size) * batch_size * 3
        k_cache_paged = torch.randn(
            num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
        )
        v_cache_paged = torch.randn(
            num_blocks, paged_kv_block_size, nheads_k, d, device=device, dtype=dtype
        )
        block_table = rearrange(
            torch.randperm(num_blocks, dtype=torch.int32, device=device),
            "(b nblocks) -> b nblocks",
            b=batch_size,
        )
        k_cache = rearrange(
            # pytorch 1.12 doesn't have indexing with int32
            k_cache_paged[block_table.to(dtype=torch.long).flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
        v_cache = rearrange(
            v_cache_paged[block_table.to(dtype=torch.long).flatten()],
            "(b nblocks) block_size ... -> b (nblocks block_size) ...",
            b=batch_size,
        )[:, :seqlen_k]
        return k_cache, v_cache, block_table, k_cache_paged, v_cache_paged, num_blocks
    
    def test_cuda_paged_attention_decoding(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model(batch_size=2, total_seqlen=0, hidden_size=96, slot_cnt_per_block=256, 
                                                       block_cnt_per_layer=6, block_cnt_needed_by_longest_seq=3, num_heads=6, num_kv_heads=6, head_size=16)
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
    
#        query = np.random.randn(2,96).astype(np.float16)
#        key = np.random.randn(2,96).astype(np.float16)
#        value = np.random.randn(2,96).astype(np.float16)
#        key_cache = np.zeros([6,24576]).astype(np.float16)  # 24576 = 256x6x16
#        value_cache = np.zeros([6,24576]).astype(np.float16)
#        block_tables = np.array([[0,1,2],[3,4,5]]).astype(np.int32)
#        context_lens = np.array([83, 65]).astype(np.int32)
        pdb.set_trace()
        query_2x1x6x16 = np.load('q_2x1x6x16.npy')
        key_2x1x6x16 = np.load('k_2x1x6x16.npy')
        value_2x1x6x16 = np.load('v_2x1x6x16.npy')
        key_cache_6x256x6x16 = np.load('k_cache_6x256x6x16.npy')
        value_cache_6x256x6x16 = np.load('v_cache_6x256x6x16.npy')
        block_tables = np.load('block_table_2x3.npy')   # [[2,4,1], [5,3,0]]
        context_lens = np.load('cache_seqlens_2.npy')   # [83, 65]
        query = np.reshape(query_2x1x6x16, (2, 96))
        key = np.reshape(key_2x1x6x16, (2, 96))
        value = np.reshape(value_2x1x6x16, (2, 96))
        key_cache = np.reshape(key_cache_6x256x6x16, (6, 24576))
        value_cache = np.reshape(value_cache_6x256x6x16, (6, 24576))

        slot_mappings = np.array([250, 500]).astype(np.int32)
        is_prompt = np.array([0]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'key_cache':key_cache, 'value_cache':value_cache, 'block_tables':block_tables, 'slot_mappings':slot_mappings, 'context_lens':context_lens, 'is_prompt':is_prompt})
    #    q_pt = torch.from_numpy(query.reshape(3, 127, 32, 16))
    #    k_pt = torch.from_numpy(key.reshape(3, 127, 32, 16))
    #    v_pt = torch.from_numpy(value.reshape(3, 127, 32, 16))
    #    out, attention = attention_ref(q_pt, k_pt, v_pt, causal=True, window_size=[-1, 0])
    #    y_np = np.array(y).reshape(381, 512)
    #    out_np = out.reshape(381, 512).numpy()
    #    #assert np.allclose(y_np, out_np, rtol=1e-3, atol=1e-3, equal_nan=True)
    #    print(np.allclose(y_np, out_np, rtol=1e-3, atol=1e-3, equal_nan=True))
    #    print(y_np)
    #    print(out_np)

if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
from onnxruntime_extensions import make_onnx_model
from onnxruntime_extensions import get_library_path as _get_library_path

import onnxruntime as _ort


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
    def _create_pagedattention_test_model(domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node('PagedAttention',  
                ['query', 'key', 'value', 'key_cache', 'value_cache', 'block_tables', 'slot_mappings', 'context_lens', 'is_prompt'], 
                ['attn_out'], 
                domain=domain, num_heads=32, num_kv_heads=32, head_size=16, scale=1.0)
        ]
        query = helper.make_tensor_value_info(
            'query', onnx_proto.TensorProto.FLOAT16, [87,512])
        key = helper.make_tensor_value_info(
            'key', onnx_proto.TensorProto.FLOAT16, [87,512])
        value = helper.make_tensor_value_info(
            'value', onnx_proto.TensorProto.FLOAT16, [87,512])
        key_cache = helper.make_tensor_value_info(
            'key_cache', onnx_proto.TensorProto.FLOAT16, [32,8192])
        value_cache = helper.make_tensor_value_info(
            'value_cache', onnx_proto.TensorProto.FLOAT16, [32,8192])
        block_tables = helper.make_tensor_value_info(
            'block_tables', onnx_proto.TensorProto.INT32, [5,3])
        slot_mappings = helper.make_tensor_value_info(
            'slot_mappings', onnx_proto.TensorProto.INT32, [87])
        context_lens = helper.make_tensor_value_info(
            'context_lens', onnx_proto.TensorProto.INT32, [5])
        is_prompt = helper.make_tensor_value_info(
            'is_prompt', onnx_proto.TensorProto.INT32, [1])
        attn_out = helper.make_tensor_value_info(
            'attn_out', onnx_proto.TensorProto.FLOAT16, [87,512])
        graph = helper.make_graph(nodes, 'test_paged_attention', 
                    [query, key, value, key_cache, value_cache, block_tables, slot_mappings, context_lens, is_prompt], 
                    [attn_out])
        model = make_onnx_model(graph)
        return model
        
    def test_cuda_paged_attention(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_pagedattention_test_model()
        sess = _ort.InferenceSession(onnx_model.SerializeToString(),
                                     so,
                                     providers=['CUDAExecutionProvider'])
        query = np.random.randn(87,512).astype(np.float16) # 87 is the token num of all the sequences (5+12+16+20+34)
        key = np.random.randn(87,512).astype(np.float16)
        value = np.random.randn(87,512).astype(np.float16)
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

if __name__ == "__main__":
    unittest.main()

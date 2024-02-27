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
            'query', onnx_proto.TensorProto.FLOAT16, [1,28,2560])
        key = helper.make_tensor_value_info(
            'key', onnx_proto.TensorProto.FLOAT16, [1,28,2560])
        value = helper.make_tensor_value_info(
            'value', onnx_proto.TensorProto.FLOAT16, [1,28,2560])
        past_key = helper.make_tensor_value_info(
            'past_key', onnx_proto.TensorProto.FLOAT16, [1,32,2048,80])
        past_value = helper.make_tensor_value_info(
            'past_value', onnx_proto.TensorProto.FLOAT16, [1,32,2048,80])
        seqlens_k = helper.make_tensor_value_info(
            'seqlens_k', onnx_proto.TensorProto.INT32, [1,1])
        total_seqlen = helper.make_tensor_value_info(
            'total_seqlen', onnx_proto.TensorProto.INT32, [1])
#        cos_cache = helper.make_tensor_value_info(
#            'cos_cache', onnx_proto.TensorProto.FLOAT, [])
#        sin_cache = helper.make_tensor_value_info(
#            'sin_cache', onnx_proto.TensorProto.FLOAT, [])
        attn_out = helper.make_tensor_value_info(
            'attn_out', onnx_proto.TensorProto.FLOAT16, [1,28,2560])
        present_key = helper.make_tensor_value_info(
            'present_key', onnx_proto.TensorProto.FLOAT16, [1,32,2048,80])
        present_value = helper.make_tensor_value_info(
            'present_value', onnx_proto.TensorProto.FLOAT16, [1,32,2048,80])

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
        query = np.random.randn(1,28,2560).astype(np.float16)
        key = np.random.randn(1,28,2560).astype(np.float16)
        value = np.random.randn(1,28,2560).astype(np.float16)
        past_key = np.zeros([1,32,2048,80]).astype(np.float16)
        past_value = np.zeros([1,32,2048,80]).astype(np.float16)
        seqlens_k = np.array([[27]]).astype(np.int32)
        total_seqlen = np.array([28]).astype(np.int32)
        y = sess.run(None, {'query':query, 'key':key, 'value':value, 'past_key':past_key, 'past_value':past_value, 'seqlens_k':seqlens_k, 'total_seqlen':total_seqlen})


if __name__ == "__main__":
    unittest.main()

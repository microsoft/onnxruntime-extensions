# coding: utf-8
import unittest

import unicodedata
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    get_library_path as _get_library_path)


class TestPythonOpNormalizer(unittest.TestCase):

    def _create_test_model_string_normalize(
            self, prefix, domain='ai.onnx.contrib'):
        nodes = [
            helper.make_node(
                '%sStringNormalize' % prefix, ['input_1'], ['output_1'],
                domain=domain)]

        input0 = helper.make_tensor_value_info(
            'input_1', onnx_proto.TensorProto.STRING, [None, None])
        output0 = helper.make_tensor_value_info(
            'output_1', onnx_proto.TensorProto.STRING, [None, None])

        graph = helper.make_graph(nodes, 'test0', [input0], [output0])
        model = helper.make_model(
            graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
        return model

    @classmethod
    def setUpClass(cls):
        pass

    def test_string_normlize_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = self._create_test_model_string_normalize('')
        self.assertIn('op_type: "StringNormalize"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        expected = np.array([[unicodedata.normalize(input_1[0, 0])]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()

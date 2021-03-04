# coding: utf-8
import unittest
import os
import base64
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_bert(prefix, domain='ai.onnx.contrib'):
    nodes = []
    mkv = helper.make_tensor_value_info
    nodes.append(helper.make_node(
        '%sBertTokenizer' % prefix,
        inputs=['text'],
        outputs=['out0', 'out1'],
        name='BertTokenizeOpName',
        domain='ai.onnx.contrib',
        vocab='{"A": 0, "##A": 1, "B": 2, "##BB": 3}'.encode('utf-8'),
        suffix_indicator="##",
    ))
    inputs = [
        mkv('model', onnx_proto.TensorProto.UINT8, [None]),
    ]

    graph = helper.make_graph(
        nodes, 'test0', inputs, [
            mkv('out0', onnx_proto.TensorProto.INT32, [None]),
            mkv('out1', onnx_proto.TensorProto.INT64, [None])
        ])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


class TestPythonOpBert(unittest.TestCase):

    def test_string_bert_tokenizer(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        cc_onnx_model = _create_test_model_bert('')
        self.assertIn('op_type: "BertTokenizer"', str(cc_onnx_model))
        cc_sess = _ort.InferenceSession(cc_onnx_model.SerializeToString(), so)

        inputs = dict(inputs=np.array(["A A B BB", "B BB A AA"], dtype=np.object))
        cc_txout = cc_sess.run(None, inputs)
        exp = [numpy.array([], dtype=numpy.int32),
               numpy.array([], dtype=numpy.int64)]
        for i in range(0, 2):
            assert_almost_equal(exp[i], py_txout[i])
            assert_almost_equal(exp[i], cc_txout[i])


if __name__ == "__main__":
    unittest.main()

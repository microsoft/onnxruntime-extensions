# coding: utf-8
import unittest
import json
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    get_library_path as _get_library_path)


def _create_test_model_bert(prefix, domain='ai.onnx.contrib'):
    words = ["[UNK]", "[CLS]", "[SEP]", "want", "##want",
             "##ed", "wa", "un", "runn", "##ing"]
    vocab = {w: i + 10 for i, w in enumerate(words)}
    st = json.dumps(vocab)
    print(st)
    nodes = []
    mkv = helper.make_tensor_value_info
    nodes.append(helper.make_node(
        '%sBertTokenizer' % prefix,
        inputs=['text'],
        outputs=['out0', 'out1'],
        name='BertTokenizerOpName',
        domain='ai.onnx.contrib',
        vocab=st.encode('utf-8'),
        suffix_indicator="##",
    ))
    inputs = [
        mkv('text', onnx_proto.TensorProto.STRING, [None]),
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

        inputs = dict(text=np.array(["unwanted running",
                                     "unwantedX running"], dtype=np.object))
        cc_txout = cc_sess.run(None, inputs)
        exp = [np.array([17, 14, 15, 18, 19,
                         17, 14, 15, -1, 18, 19], dtype=np.int32),
               np.array([0, 5, 11], dtype=np.int64)]
        for i in range(0, 2):
            assert_almost_equal(exp[i], cc_txout[i])


if __name__ == "__main__":
    unittest.main()

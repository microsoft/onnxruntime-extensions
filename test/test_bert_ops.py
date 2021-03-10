# coding: utf-8
import unittest
import json
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    get_library_path as _get_library_path)


def _create_test_model_wordpiece(prefix, domain='ai.onnx.contrib'):
    words = ["want", "##want",
             "##ed", "wa", "un", "runn", "##ing"]
    vocab = {w: i + 10 for i, w in enumerate(words)}
    st = json.dumps(vocab)
    nodes = []
    mkv = helper.make_tensor_value_info
    reg = helper.make_tensor("pattern", onnx_proto.TensorProto.STRING, [1, ],
                             ["(\\s)".encode('ascii')])
    reg_empty = helper.make_tensor("keep_pattern", onnx_proto.TensorProto.STRING, [0, ], [])

    nodes.append(helper.make_node(
        '%sStringRegexSplitWithOffsets' % prefix,
        inputs=['text', 'pattern', 'keep_pattern'],
        outputs=['words', 'begin', 'end', 'rows'],
        name='StringRegexSplitOpName',
        domain='ai.onnx.contrib'
    ))
    nodes.append(helper.make_node(
        '%sWordpieceTokenizer' % prefix,
        inputs=['words', 'rows'],
        outputs=['out0', 'out1', 'out2', 'out3'],
        name='BertTokenizerOpName',
        domain='ai.onnx.contrib',
        vocab=st.encode('utf-8'),
        suffix_indicator="##",
        unknown_token="[UNK]",
    ))

    inputs = [
        mkv('text', onnx_proto.TensorProto.STRING, [None]),
    ]
    graph = helper.make_graph(
        nodes, 'test0', inputs, [
            mkv('out0', onnx_proto.TensorProto.STRING, [None]),
            mkv('out1', onnx_proto.TensorProto.INT64, [None]),
            mkv('out2', onnx_proto.TensorProto.INT64, [None]),
            mkv('out3', onnx_proto.TensorProto.INT64, [None]),
            mkv('words', onnx_proto.TensorProto.STRING, [None]),
            mkv('rows', onnx_proto.TensorProto.INT64, [None])],
        [reg, reg_empty]
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


class TestPythonOpBert(unittest.TestCase):

    def test_string_wordpiece_tokenizer(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        cc_onnx_model = _create_test_model_wordpiece('')
        self.assertIn('op_type: "WordpieceTokenizer"', str(cc_onnx_model))
        cc_sess = _ort.InferenceSession(cc_onnx_model.SerializeToString(), so)

        inputs = dict(text=np.array(["unwanted running",
                                     "unwantedX running"], dtype=np.object))
        cc_txout = cc_sess.run(None, inputs)
        exp = [np.array(['un', '##want', '##ed', 'runn', '##ing',
                         'un', '##want', '##ed', '[UNK]', 'runn', '##ing']),
               np.array([14, 11, 12, 15, 16, 14, 11, 12,
                         -1, 15, 16], dtype=np.int32),
               np.array([0, 5, 11], dtype=np.int64),
               np.array(['unwanted', 'running', 'unwantedX', 'running']),
               np.array([0, 2, 4], dtype=np.int64)]
        for i in range(0, 5):
            try:
                assert_almost_equal(exp[i], cc_txout[i])
            except TypeError:
                assert exp[i].tolist() == cc_txout[i].tolist()


if __name__ == "__main__":
    unittest.main()

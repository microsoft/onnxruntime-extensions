# coding: utf-8
import unittest
import os
import base64
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_extensions import (
    util,
    onnx_op,
    PyCustomOpDef,
    OrtPyFunction,
    make_onnx_model,
    get_library_path as _get_library_path)
import tensorflow as tf
from tensorflow_text import SentencepieceTokenizer


def load_piece(name):
    fullname = os.path.join(
        os.path.dirname(__file__), "data",
        "%s_%s.txt" % (
            os.path.splitext(os.path.split(__file__)[-1])[0],
            name))
    with open(fullname, "r") as f:
        content = f.read()
    t = base64.decodebytes(content.encode())
    b64 = base64.b64encode(t)
    return np.array(list(t), dtype=np.uint8), b64


def _create_test_model_sentencepiece(
        prefix, model_b64, domain='ai.onnx.contrib'):
    nodes = []
    mkv = helper.make_tensor_value_info
    if model_b64 is None:
        nodes.append(helper.make_node(
            '%sSentencepieceTokenizer' % prefix,
            inputs=[
                'model',  # model__6
                'inputs',  # inputs
                'nbest_size',
                'alpha',
                'add_bos',
                'add_eos',
                'reverse',
            ],
            outputs=['out0', 'out1'],
            name='SentencepieceTokenizeOpName',
            domain=domain,
        ))
        inputs = [
            mkv('model', onnx_proto.TensorProto.UINT8, [None]),
            mkv('inputs', onnx_proto.TensorProto.STRING, [None]),
            mkv('nbest_size', onnx_proto.TensorProto.INT64, [None]),
            mkv('alpha', onnx_proto.TensorProto.FLOAT, [None]),
            mkv('add_bos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('add_eos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('reverse', onnx_proto.TensorProto.BOOL, [None])
        ]
    else:
        nodes.append(helper.make_node(
            '%sSentencepieceTokenizer' % prefix,
            inputs=[
                'inputs',  # inputs
                'nbest_size',
                'alpha',
                'add_bos',
                'add_eos',
                'reverse',
            ],
            outputs=['out0', 'out1'],
            model=model_b64,
            name='SentencepieceTokenizeOpName',
            domain='ai.onnx.contrib',
        ))
        inputs = [
            mkv('inputs', onnx_proto.TensorProto.STRING, [None]),
            mkv('nbest_size', onnx_proto.TensorProto.INT64, [None]),
            mkv('alpha', onnx_proto.TensorProto.FLOAT, [None]),
            mkv('add_bos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('add_eos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('reverse', onnx_proto.TensorProto.BOOL, [None])
        ]

    graph = helper.make_graph(
        nodes, 'test0', inputs, [
            mkv('out0', onnx_proto.TensorProto.INT32, [None]),
            mkv('out1', onnx_proto.TensorProto.INT64, [None])
        ])
    model = make_onnx_model(graph)
    return model


def _create_test_model_ragged_to_sparse(
        prefix, model_b64, domain='ai.onnx.contrib'):
    nodes = []
    mkv = helper.make_tensor_value_info
    if model_b64 is None:
        nodes.append(helper.make_node(
            '%sSentencepieceTokenizer' % prefix,
            inputs=[
                'model',  # model__6
                'inputs',  # inputs
                'nbest_size',
                'alpha',
                'add_bos',
                'add_eos',
                'reverse',
            ],
            outputs=['tokout0', 'tokout1'],
            name='SentencepieceTokenizeOpName',
            domain=domain,
        ))
        inputs = [
            mkv('model', onnx_proto.TensorProto.UINT8, [None]),
            mkv('inputs', onnx_proto.TensorProto.STRING, [None]),
            mkv('nbest_size', onnx_proto.TensorProto.INT64, [None]),
            mkv('alpha', onnx_proto.TensorProto.FLOAT, [None]),
            mkv('add_bos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('add_eos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('reverse', onnx_proto.TensorProto.BOOL, [None])
        ]

        nodes.append(helper.make_node(
            '%sRaggedTensorToSparse' % prefix,
            inputs=['tokout1', 'tokout0'],
            outputs=['out0', 'out1', 'out2'],
            name='RaggedTensorToSparse',
            domain='ai.onnx.contrib',
        ))
    else:
        nodes.append(helper.make_node(
            '%sSentencepieceTokenizer' % prefix,
            inputs=[
                'inputs',  # inputs
                'nbest_size',
                'alpha',
                'add_bos',
                'add_eos',
                'reverse',
            ],
            outputs=['tokout0', 'tokout1'],
            model=model_b64,
            name='SentencepieceTokenizeOpName',
            domain='ai.onnx.contrib',
        ))
        inputs = [
            mkv('inputs', onnx_proto.TensorProto.STRING, [None]),
            mkv('nbest_size', onnx_proto.TensorProto.INT64, [None]),
            mkv('alpha', onnx_proto.TensorProto.FLOAT, [None]),
            mkv('add_bos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('add_eos', onnx_proto.TensorProto.BOOL, [None]),
            mkv('reverse', onnx_proto.TensorProto.BOOL, [None])
        ]

        nodes.append(helper.make_node(
            'Shape', inputs=['tokout1'], outputs=['n_els']))

        nodes.append(helper.make_node(
            'RaggedTensorToSparse',
            inputs=['tokout1'],
            outputs=['out0', 'out2'],
            name='RaggedTensorToSparse',
            domain='ai.onnx.contrib',
        ))

        nodes.append(helper.make_node(
            'Identity', inputs=['tokout0'], outputs=['out1']))

    graph = helper.make_graph(
        nodes, 'test0', inputs, [
            mkv('out0', onnx_proto.TensorProto.INT64, [None]),
            mkv('out1', onnx_proto.TensorProto.INT32, [None]),
            mkv('out2', onnx_proto.TensorProto.INT64, [None])
        ])
    model = make_onnx_model(graph)
    return model


def _create_test_model_ragged_to_dense(
        prefix, model_b64, domain='ai.onnx.contrib'):
    nodes = []
    mkv = helper.make_tensor_value_info
    nodes.append(helper.make_node(
        '%sSentencepieceTokenizer' % prefix,
        inputs=[
            'inputs',  # inputs
            'nbest_size',
            'alpha',
            'add_bos',
            'add_eos',
            'reverse',
        ],
        outputs=['tokout0', 'tokout1'],
        model=model_b64,
        name='SentencepieceTokenizeOpName',
        domain=domain,
    ))
    inputs = [
        mkv('inputs', onnx_proto.TensorProto.STRING, [None]),
        mkv('nbest_size', onnx_proto.TensorProto.INT64, [None]),
        mkv('alpha', onnx_proto.TensorProto.FLOAT, [None]),
        mkv('add_bos', onnx_proto.TensorProto.BOOL, [None]),
        mkv('add_eos', onnx_proto.TensorProto.BOOL, [None]),
        mkv('reverse', onnx_proto.TensorProto.BOOL, [None])
    ]

    nodes.append(helper.make_node(
        'Shape', inputs=['tokout1'], outputs=['n_els']))
    nodes.append(helper.make_node(
        'Cast', inputs=['tokout0'], outputs=['tokout064'], to=onnx_proto.TensorProto.INT64))

    default_value = helper.make_tensor("default_value", onnx_proto.TensorProto.INT64, [1, ], [-1])
    unused = helper.make_tensor("unused", onnx_proto.TensorProto.INT64, [0, ], [])

    nodes.append(helper.make_node(
        '%sRaggedTensorToDense' % prefix,
        inputs=['unused', 'tokout064', 'default_value', 'tokout1'],
        outputs=['out0'],
        name='RaggedTensorToDense',
        domain='ai.onnx.contrib',
    ))

    nodes.append(helper.make_node(
        'Identity', inputs=['tokout0'], outputs=['out1']))

    graph = helper.make_graph(
        nodes, 'test0', inputs, [
            mkv('out0', onnx_proto.TensorProto.INT64, [None]),
            mkv('out1', onnx_proto.TensorProto.INT32, [None]),
        ], [default_value, unused])
    model = make_onnx_model(graph)
    return model


class TestPythonOpSentencePiece(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="PySentencepieceTokenizer",
                 inputs=[PyCustomOpDef.dt_uint8,  # 0: input,
                         PyCustomOpDef.dt_string,  # 1: input
                         PyCustomOpDef.dt_int64,  # 2: nbest_size
                         PyCustomOpDef.dt_float,  # 3: alpha
                         PyCustomOpDef.dt_bool,  # 4: add_bos
                         PyCustomOpDef.dt_bool,  # 5: add_eos
                         PyCustomOpDef.dt_bool],  # 6: reverse
                 outputs=[PyCustomOpDef.dt_int32,
                          PyCustomOpDef.dt_int64])
        def sentence_piece_tokenizer_op(model, inputs, nbest_size,
                                        alpha, add_bos, add_eos, reverse):
            """Implements `text.SentencepieceTokenizer
            <https://github.com/tensorflow/text/blob/master/docs/
            api_docs/python/text/SentencepieceTokenizer.md>`_."""
            # The custom op implementation.
            tokenizer = SentencepieceTokenizer(
                model=model.tobytes(),
                reverse=reverse[0],
                add_bos=add_bos[0],
                add_eos=add_eos[0],
                alpha=alpha[0],
                nbest_size=nbest_size[0])
            ragged_tensor = tokenizer.tokenize(inputs)
            output_values = ragged_tensor.flat_values.numpy()
            output_splits = ragged_tensor.nested_row_splits[0].numpy()
            return output_values, output_splits

        cls.SentencepieceTokenizer = sentence_piece_tokenizer_op

        @onnx_op(op_type="PyRaggedTensorToSparse",
                 inputs=[PyCustomOpDef.dt_int64,
                         PyCustomOpDef.dt_int32],
                 outputs=[PyCustomOpDef.dt_int64,
                          PyCustomOpDef.dt_int32,
                          PyCustomOpDef.dt_int64])
        def ragged_tensor_to_sparse(nested_splits, dense_values):
            sparse_indices, sparse_values, sparse_dense_shape = \
                tf.raw_ops.RaggedTensorToSparse(
                    rt_nested_splits=[nested_splits],
                    rt_dense_values=dense_values)
            return (sparse_indices.numpy(),
                    sparse_values.numpy(),
                    sparse_dense_shape.numpy())

        cls.RaggedTensorToSparse = ragged_tensor_to_sparse

    def test_string_ragged_string_to_sparse_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        model, model_b64 = load_piece('model__6')
        onnx_model = _create_test_model_ragged_to_sparse('Py', None)
        self.assertIn('op_type: "PyRaggedTensorToSparse"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        inputs = dict(
            model=model,
            inputs=np.array(
                ["Hello world", "Hello world louder"], dtype=np.object),
            nbest_size=np.array([0], dtype=np.int64),
            alpha=np.array([0], dtype=np.float32),
            add_bos=np.array([0], dtype=np.bool_),
            add_eos=np.array([0], dtype=np.bool_),
            reverse=np.array([0], dtype=np.bool_))
        txout = sess.run(None, inputs)
        temp = self.SentencepieceTokenizer(**inputs)
        exp = self.RaggedTensorToSparse(temp[1], temp[0])
        for i in range(0, 3):
            assert_almost_equal(exp[i], txout[i])

    def test_string_ragged_string_to_sparse_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        model, model_b64 = load_piece('model__6')
        onnx_model = _create_test_model_ragged_to_sparse('', model_b64)
        self.assertIn('op_type: "RaggedTensorToSparse"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        inputs = dict(
            model=model,
            inputs=np.array(
                ["Hello world", "Hello world louder"], dtype=np.object),
            nbest_size=np.array([0], dtype=np.int64),
            alpha=np.array([0], dtype=np.float32),
            add_bos=np.array([0], dtype=np.bool_),
            add_eos=np.array([0], dtype=np.bool_),
            reverse=np.array([0], dtype=np.bool_))
        temp = self.SentencepieceTokenizer(**inputs)
        exp = self.RaggedTensorToSparse(temp[1], temp[0])
        del inputs['model']
        txout = sess.run(None, inputs)
        assert_almost_equal(exp[0], txout[0])
        assert_almost_equal(exp[1], txout[1])
        assert_almost_equal(exp[2], txout[2])

    def test_string_ragged_string_to_dense_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        model, model_b64 = load_piece('model__6')
        onnx_model = _create_test_model_ragged_to_dense('', model_b64)
        self.assertIn('op_type: "RaggedTensorToDense"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        inputs = dict(
            model=model,
            inputs=np.array(
                ["Hello world", "Hello world louder"], dtype=np.object),
            nbest_size=np.array([0], dtype=np.int64),
            alpha=np.array([0], dtype=np.float32),
            add_bos=np.array([0], dtype=np.bool_),
            add_eos=np.array([0], dtype=np.bool_),
            reverse=np.array([0], dtype=np.bool_))
        del inputs['model']
        txout = sess.run(None, inputs)
        assert_almost_equal(
            txout[0], np.array([[17486, 1017, -1, -1], [17486, 1017, 155, 21869]], dtype=np.int64))
        assert_almost_equal(
            txout[1], np.array([17486, 1017, 17486, 1017, 155, 21869], dtype=np.int32))

    def test_string_sentencepiece_tokenizer(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        model, model_b64 = load_piece('model__6')
        py_onnx_model = _create_test_model_sentencepiece('Py', None)
        self.assertIn(
            'op_type: "PySentencepieceTokenizer"', str(py_onnx_model))
        cc_onnx_model = _create_test_model_sentencepiece('', model_b64)
        self.assertIn('op_type: "SentencepieceTokenizer"', str(cc_onnx_model))
        py_sess = _ort.InferenceSession(py_onnx_model.SerializeToString(), so)
        cc_sess = _ort.InferenceSession(cc_onnx_model.SerializeToString(), so)

        for alpha in [0, 0.5]:
            for nbest_size in [0, 1]:
                for bools in range(0, 8):
                    with self.subTest(
                            alpha=alpha, nbest_size=nbest_size, bools=bools):
                        inputs = dict(
                            model=model,
                            inputs=np.array(
                                ["Hello world", "Hello world louder"],
                                dtype=np.object),
                            nbest_size=np.array(
                                [nbest_size], dtype=np.int64),
                            alpha=np.array([alpha], dtype=np.float32),
                            add_bos=np.array([bools & 1], dtype=np.bool_),
                            add_eos=np.array([bools & 2], dtype=np.bool_),
                            reverse=np.array([bools & 4], dtype=np.bool_))
                        exp = self.SentencepieceTokenizer(**inputs)
                        py_txout = py_sess.run(None, inputs)
                        del inputs['model']
                        cc_txout = cc_sess.run(None, inputs)
                        for i in range(0, 2):
                            assert_almost_equal(exp[i], py_txout[i])
                            assert_almost_equal(exp[i], cc_txout[i])

    def test_string_sentencepiece_tokenizer_bin(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        model, model_b64 = load_piece('model__6')
        modelb = bytes(model)
        py_onnx_model = _create_test_model_sentencepiece('Py', None)
        self.assertIn(
            'op_type: "PySentencepieceTokenizer"', str(py_onnx_model))
        cc_onnx_model = _create_test_model_sentencepiece('', modelb)
        self.assertIn('op_type: "SentencepieceTokenizer"', str(cc_onnx_model))
        py_sess = _ort.InferenceSession(py_onnx_model.SerializeToString(), so)
        cc_sess = _ort.InferenceSession(cc_onnx_model.SerializeToString(), so)

        alpha = 0
        nbest_size = 0
        bools = 0
        inputs = dict(
            model=model,
            inputs=np.array(
                ["Hello world", "Hello world louder"],
                dtype=np.object),
            nbest_size=np.array(
                [nbest_size], dtype=np.int64),
            alpha=np.array([alpha], dtype=np.float32),
            add_bos=np.array([bools & 1], dtype=np.bool_),
            add_eos=np.array([bools & 2], dtype=np.bool_),
            reverse=np.array([bools & 4], dtype=np.bool_))
        exp = self.SentencepieceTokenizer(**inputs)
        py_txout = py_sess.run(None, inputs)
        del inputs['model']
        cc_txout = cc_sess.run(None, inputs)
        for i in range(0, 2):
            assert_almost_equal(exp[i], py_txout[i])
            assert_almost_equal(exp[i], cc_txout[i])

    def test_external_pretrained_model(self):
        fullname = util.get_test_data_file('data', 'en.wiki.bpe.vs100000.model')
        ofunc = OrtPyFunction.from_customop('SentencepieceTokenizer', model=open(fullname, 'rb').read())

        alpha = 0
        nbest_size = 0
        flags = 0
        tokens, indices = ofunc(
            np.array(['best hotel in bay area.']),
            np.array(
                [nbest_size], dtype=np.int64),
            np.array([alpha], dtype=np.float32),
            np.array([flags & 1], dtype=np.bool_),
            np.array([flags & 2], dtype=np.bool_),
            np.array([flags & 4], dtype=np.bool_))
        self.assertEqual(tokens.tolist(), [1095, 4054, 26, 2022, 755, 99935])


    def test_spm_decoder(self):
        fullname = util.get_test_data_file('data', 'en.wiki.bpe.vs100000.model')
        ofunc = OrtPyFunction.from_customop('SentencepieceDecoder', model=open(fullname, 'rb').read())

        result = ofunc(np.array([1095, 4054, 26, 2022, 755, 99935], dtype=np.int64))
        self.assertEqual(' '.join(result), 'best hotel in bay area.')


if __name__ == "__main__":
    unittest.main()

# coding: utf-8
import io
import json
import sys
import unittest
import re
from binascii import crc32
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path,
    hash_64)

NUM_BUCKETS = 23


def _create_test_model_string_upper(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('%sStringUpper' % prefix,
                                  ['identity1'], ['customout'],
                                  domain=domain)]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None, None])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None, None])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_lower(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('%sStringLower' % prefix,
                                  ['identity1'], ['customout'],
                                  domain=domain)]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None, None])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None, None])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_join(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(
        helper.make_node('Identity', ['text'], ['identity1']))
    nodes.append(
        helper.make_node('Identity', ['sep'], ['identity2']))
    nodes.append(
        helper.make_node('Identity', ['axis'], ['identity3']))
    nodes.append(
        helper.make_node(
            '%sStringJoin' % prefix, ['identity1', 'identity2', 'identity3'],
            ['customout'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'text', onnx_proto.TensorProto.STRING, None)
    input1 = helper.make_tensor_value_info(
        'sep', onnx_proto.TensorProto.STRING, [1])
    input2 = helper.make_tensor_value_info(
        'axis', onnx_proto.TensorProto.INT64, [1])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, None)

    graph = helper.make_graph(
        nodes, 'test0', [input0, input1, input2], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_replace(prefix, domain='ai.onnx.contrib',
                                      global_replace=True):
    nodes = []
    nodes.append(
        helper.make_node('Identity', ['text'], ['id1']))
    nodes.append(
        helper.make_node('Identity', ['pattern'], ['id2']))
    nodes.append(
        helper.make_node('Identity', ['rewrite'], ['id3']))
    if global_replace:
        nodes.append(
            helper.make_node(
                '%sStringRegexReplace' % prefix, ['id1', 'id2', 'id3'],
                ['customout'], domain=domain))
    else:
        nodes.append(
            helper.make_node(
                '%sStringRegexReplace' % prefix, ['id1', 'id2', 'id3'],
                ['customout'], domain=domain,
                global_replace=0))

    input0 = helper.make_tensor_value_info(
        'text', onnx_proto.TensorProto.STRING, [None, 1])
    input1 = helper.make_tensor_value_info(
        'pattern', onnx_proto.TensorProto.STRING, [1])
    input2 = helper.make_tensor_value_info(
        'rewrite', onnx_proto.TensorProto.STRING, [1])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None, 1])

    graph = helper.make_graph(
        nodes, 'test0', [input0, input1, input2], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_to_hash(
        prefix, domain='ai.onnx.contrib', kind=None):
    if kind == 'crc32':
        op_type = 'StringToCRC32'
        out_type = onnx_proto.TensorProto.UINT32
        in_type = out_type
    elif kind == 'hash_bucket':
        op_type = 'StringToHashBucket'
        out_type = onnx_proto.TensorProto.INT64
        in_type = out_type
    elif kind == 'hash_bucket_fast':
        op_type = 'StringToHashBucketFast'
        out_type = onnx_proto.TensorProto.INT64
        in_type = out_type
    else:
        raise ValueError('Unknown value %r.' % kind)
    nodes = []
    nodes.append(
        helper.make_node('Identity', ['text'], ['id1']))
    nodes.append(
        helper.make_node('Identity', ['num_buckets'], ['id2']))
    nodes.append(
        helper.make_node(
            '%s%s' % (prefix, op_type), ['id1', 'id2'],
            ['customout'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'text', onnx_proto.TensorProto.STRING, [None, None])
    input1 = helper.make_tensor_value_info(
        'num_buckets', in_type, [1])
    output0 = helper.make_tensor_value_info(
        'customout', out_type, [None, None])

    graph = helper.make_graph(
        nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_equal(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(helper.make_node('Identity', ['x'], ['id1']))
    nodes.append(helper.make_node('Identity', ['y'], ['id2']))
    nodes.append(
        helper.make_node(
            '%sStringEqual' % prefix, ['id1', 'id2'], ['z'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'x', onnx_proto.TensorProto.STRING, [])
    input1 = helper.make_tensor_value_info(
        'y', onnx_proto.TensorProto.STRING, [])
    output0 = helper.make_tensor_value_info(
        'z', onnx_proto.TensorProto.BOOL, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_split(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(helper.make_node('Identity', ['input'], ['id1']))
    nodes.append(helper.make_node('Identity', ['delimiter'], ['id2']))
    nodes.append(helper.make_node('Identity', ['skip_empty'], ['id3']))
    nodes.append(
        helper.make_node(
            '%sStringSplit' % prefix, ['id1', 'id2', 'id3'],
            ['indices', 'values', 'shape'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'input', onnx_proto.TensorProto.STRING, [])
    input1 = helper.make_tensor_value_info(
        'delimiter', onnx_proto.TensorProto.STRING, [])
    input2 = helper.make_tensor_value_info(
        'skip_empty', onnx_proto.TensorProto.BOOL, [])
    output0 = helper.make_tensor_value_info(
        'indices', onnx_proto.TensorProto.INT64, [])
    output1 = helper.make_tensor_value_info(
        'values', onnx_proto.TensorProto.STRING, [])
    output2 = helper.make_tensor_value_info(
        'shape', onnx_proto.TensorProto.INT64, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1, input2],
                              [output0, output1, output2])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_regex_split(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(helper.make_node('Identity', ['input'], ['id1']))
    nodes.append(helper.make_node('Identity', ['pattern'], ['id2']))
    nodes.append(helper.make_node('Identity', ['keep_pattern'], ['id3']))
    nodes.append(
        helper.make_node(
            '%sStringRegexSplitWithOffsets' % prefix, ['id1', 'id2', 'id3'],
            ['tokens', 'begins', 'ends', 'row_indices'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'input', onnx_proto.TensorProto.STRING, [])
    input1 = helper.make_tensor_value_info(
        'pattern', onnx_proto.TensorProto.STRING, [])
    input2 = helper.make_tensor_value_info(
        'keep_pattern', onnx_proto.TensorProto.STRING, [])
    output0 = helper.make_tensor_value_info(
        'tokens', onnx_proto.TensorProto.STRING, [])
    output1 = helper.make_tensor_value_info(
        'begins', onnx_proto.TensorProto.INT64, [])
    output2 = helper.make_tensor_value_info(
        'ends', onnx_proto.TensorProto.INT64, [])
    output3 = helper.make_tensor_value_info(
        'row_indices', onnx_proto.TensorProto.INT64, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1, input2],
                              [output0, output1, output2, output3])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


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

class TestPythonOpString(unittest.TestCase):

    _string_join = None
    _string_to_crc32 = None

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="PyStringUpper",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_upper(x):
            # The user custom op implementation here.
            return np.array([s.upper() for s in x.ravel()]).reshape(x.shape)

        @onnx_op(op_type="PyStringLower",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_lower(x):
            # The user custom op implementation here.
            return np.array([s.lower() for s in x.ravel()]).reshape(x.shape)

        @onnx_op(op_type="PyStringJoin",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_string,
                         PyCustomOpDef.dt_int64],
                 outputs=[PyCustomOpDef.dt_string])
        def string_join(x, sep, axis):
            # The user custom op implementation here.
            if sep.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'sep'.".format(sep.shape))
            if axis.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'axis'.".format(axis.shape))
            sp = sep[0]
            ax = axis[0]
            if ax < 0 or ax >= len(x.shape):
                raise RuntimeError(
                    "axis must be in [%r,%r] but is %r" % (
                        0, len(x.shape), ax))
            if len(x.shape) == 1:
                return np.array([sp.join(x)])
            dims = np.arange(len(x.shape))
            dims[ax], dims[-1] = dims[-1], dims[ax]
            x2 = np.transpose(x, dims)
            res_shape = x2.shape[:-1]
            x2 = x2.reshape((-1, x2.shape[-1]))
            res = np.empty(x2.shape[0], dtype=x.dtype)
            for i in range(x2.shape[0]):
                res[i] = sp.join(x2[i, :])
            return res.reshape(res_shape)

        @onnx_op(op_type="PyStringRegexReplace",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_string,
                         PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_replace(x, pattern, rewrite):
            # The user custom op implementation here.
            if pattern.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'pattern'.".format(pattern.shape))
            if rewrite.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'rewrite'.".format(rewrite.shape))
            reg = re.compile(pattern[0])
            res = np.array(
                list(map(lambda t: reg.sub(rewrite[0], t), x.ravel())))
            return res.reshape(x.shape)

        @onnx_op(op_type="PyStringToCRC32",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_uint32],
                 outputs=[PyCustomOpDef.dt_uint32])
        def string_to_crc32(x, num_buckets):
            if num_buckets.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'num_buckets'.".format(
                        num_buckets.shape))
            nb = num_buckets[0]
            res = np.array(
                list(map(
                    lambda x: crc32(x.encode('iso-8859-15')) % nb,
                    x.ravel())))
            return res.reshape(x.shape)

        @onnx_op(op_type="PyStringToHashBucket",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_int64],
                 outputs=[PyCustomOpDef.dt_int64])
        def string_to_hash_bucket(x, num_buckets):
            if num_buckets.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'num_buckets'.".format(
                        num_buckets.shape))
            nb = num_buckets[0]
            res = np.array(
                list(map(lambda x: hash_64(x, nb, True), x.ravel())))
            return res.reshape(x.shape).astype(np.int64)

        @onnx_op(op_type="PyStringEqual",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_bool])
        def string_equal(x, y):
            return x == y

        @onnx_op(op_type="PyStringSplit",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_string,
                         PyCustomOpDef.dt_bool],
                 outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_string,
                          PyCustomOpDef.dt_int64])
        def string_split(input, delimiter, skip_empty):
            if delimiter.shape != (1, ):
                raise RuntimeError("demiliter must a single element tensor.")
            if skip_empty.shape != (1, ):
                raise RuntimeError("skip_empty must a single element tensor.")
            if len(input.shape) != 1:
                raise RuntimeError("input must a one dimension tensor.")
            delimiter = delimiter[0]
            skip_empty = skip_empty[0]
            texts = []
            indices = []
            max_split = 0
            for row, text in enumerate(input):
                if not text:
                    continue
                res = text.split(delimiter)
                if skip_empty:
                    res = [t for t in res if t]
                texts.extend(res)
                max_split = max(max_split, len(res))
                indices.extend((row, i) for i in range(len(res)))
            return (np.array(indices, dtype=np.int64),
                    np.array(texts),
                    np.array([len(input), max_split], dtype=np.int64))

        cls._string_join = string_join
        cls._string_to_crc32 = string_to_crc32

    def test_check_types(self):
        def_list = set(dir(PyCustomOpDef))
        type_list = [
            # 'dt_bfloat16',
            'dt_bool',
            'dt_complex128',
            'dt_complex64',
            'dt_double',
            'dt_float',
            'dt_float16',
            'dt_int16',
            'dt_int32',
            'dt_int64',
            'dt_int8',
            'dt_string',
            'dt_uint16',
            'dt_uint32',
            'dt_uint64',
            'dt_uint8']
        for t in type_list:
            self.assertIn(t, def_list)

    def test_string_upper_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABC"]]).tolist())

    def test_string_lower_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_lower('')
        self.assertIn('op_type: "StringLower"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["abc"]]).tolist())

    def test_string_upper_cc_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["R"], ["Abcé"], ["ABC"], ["A"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(
            txout[0].tolist(),
            np.array([["R"], ["ABCé"], ["ABC"], ["A"]]).tolist())

    def test_string_lower_cc_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_lower('')
        self.assertIn('op_type: "StringLower"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["R"], ["Abce"], ["ABC"], ["A"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(
            txout[0].tolist(),
            np.array([["r"], ["abce"], ["abc"], ["a"]]).tolist())
        input_1 = np.array([['漢'], ["Abcé"]])
        try:
            txout = sess.run(None, {'input_1': input_1})
        except UnicodeDecodeError as e:
            if sys.platform == 'win32':
                # onnxruntime is not compiled on Windows
                # with utf-8 enabled.
                return
            raise e
        self.assertEqual(
            txout[0].tolist(),
            np.array([['漢'], ["abcé"]]).tolist())

    def test_string_upper_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('Py')
        self.assertIn('op_type: "PyStringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABC"]]).tolist())

    def test_string_lower_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_lower('Py')
        self.assertIn('op_type: "PyStringLower"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["abc"]]).tolist())

    def test_string_upper_python_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('Py')
        self.assertIn('op_type: "PyStringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abcé"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(),
                         np.array([["ABCé".upper()]]).tolist())

    def test_string_lower_python_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_lower('Py')
        self.assertIn('op_type: "PyStringLower"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abcé"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(),
                         np.array([["abcé".lower()]]).tolist())

    def test_string_join_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('Py')
        self.assertIn('op_type: "PyStringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.vstack([np.array([["a", "b", "c"]]),
                          np.array([["aa", "bb", ""]])])
        self.assertEqual(text.shape, (2, 3))
        sep = np.array([";"])
        axis = np.array([1], dtype=np.int64)
        TestPythonOpString._string_join(text, sep, axis)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;c", "aa;bb;"]).tolist())
        axis = np.array([0], dtype=np.int64)
        TestPythonOpString._string_join(text, sep, axis)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array(['a;aa', 'b;bb', 'c;']).tolist())

    def test_string_join_python_3d(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('Py')
        self.assertIn('op_type: "PyStringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.vstack([np.array([["a", "b", "c"]]),
                          np.array([["aa", "bb", ""]])]).reshape((2, 3, 1))
        sep = np.array([";"])
        axis = np.array([1], dtype=np.int64)
        TestPythonOpString._string_join(text, sep, axis)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array([['a;b;c'], ['aa;bb;']]).tolist())

    def test_string_join_python_1d(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('Py')
        self.assertIn('op_type: "PyStringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.array(["a", "b", "cc"])
        sep = np.array([";"])
        axis = np.array([0], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(txout[0].shape, (1, ))
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;cc"]).tolist())

    def test_string_join_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('')
        self.assertIn('op_type: "StringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.vstack([np.array([["a", "b", "c"]]),
                          np.array([["aa", "bb", ""]])])
        sep = np.array([";"])
        axis = np.array([1], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;c", "aa;bb;"]).tolist())
        axis = np.array([0], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array(['a;aa', 'b;bb', 'c;']).tolist())

    def test_string_join_cc_1d(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('')
        self.assertIn('op_type: "StringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.array(["a", "b", "cc"])
        sep = np.array([";"])
        axis = np.array([0], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;cc"]).tolist())

    def test_string_join_cc_3d(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('')
        self.assertIn('op_type: "StringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.array(["a", "b", "c", "d", "e", "f", "g", "h"]).reshape((
            2, 2, 2))
        sep = np.array([";"])
        axis = np.array([2], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(),
            np.array([['a;b', 'c;d'], ['e;f', 'g;h']]).tolist())
        axis = np.array([1], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(),
            np.array([['a;c', 'b;d'], ['e;g', 'f;h']]).tolist())
        axis = np.array([0], dtype=np.int64)
        txout = sess.run(None, {'text': text, 'sep': sep, 'axis': axis})
        self.assertEqual(
            txout[0].tolist(),
            np.array([['a;e', 'b;f'], ['c;g', 'd;h']]).tolist())

    def test_string_replace_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('')
        self.assertIn('op_type: "StringRegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_\1(void) {'])
        text = np.array([['def myfunc():'], ['def dummy():']])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject* py_myfunc(void) {'],
               ['static PyObject* py_dummy(void) {']]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_replace_cc_first(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace(
            '', global_replace=False)
        self.assertIn('op_type: "StringRegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_\1(void) {'])
        text = np.array([['def myfunc():def myfunc():'],
                         ['def dummy():def dummy():']])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject* py_myfunc(void) {def myfunc():'],
               ['static PyObject* py_dummy(void) {def dummy():']]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_replace_cc_x2(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('')
        self.assertIn('op_type: "StringRegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_\1(void) {'])
        text = np.array([['def myfunc():'], ['def dummy():' * 2]])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject* py_myfunc(void) {'],
               ['static PyObject* py_dummy(void) {' * 2]]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_replace_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('Py')
        self.assertIn('op_type: "PyStringRegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject*\npy_\1(void)\n{'])
        text = np.array([['def myfunc():'], ['def dummy():']])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject*\npy_myfunc(void)\n{'],
               ['static PyObject*\npy_dummy(void)\n{']]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_replace_python_x2(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('Py')
        self.assertIn('op_type: "PyStringRegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject*\npy_\1(void)\n{'])
        text = np.array([['def myfunc():'], ['def dummy():' * 2]])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject*\npy_myfunc(void)\n{'],
               ['static PyObject*\npy_dummy(void)\n{' * 2]]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_to_crc32_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_to_hash('Py', kind='crc32')
        self.assertIn('op_type: "PyStringToCRC32"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.array([["abc", "abcdé"], ["$$^l!%*ù", ""]])
        num_buckets = np.array([44], dtype=np.uint32)
        res = self._string_to_crc32(text, num_buckets)
        self.assertEqual(res.shape, text.shape)
        exp = np.array([[10, 38], [29, 0]], dtype=np.uint32)
        self.assertEqual(exp.tolist(), res.tolist())
        txout = sess.run(
            None, {'text': text, 'num_buckets': num_buckets})
        self.assertEqual(exp.tolist(), txout[0].tolist())

    def test_string_to_hash_bucket_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_to_hash(
            '', kind='hash_bucket')
        self.assertIn('op_type: "StringToHashBucket"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        raw = ["abc", "abcdé", "$$^l!%*ù", "", "a", "A"]
        text = np.array(raw).reshape((3, 2))
        num_buckets = np.array([NUM_BUCKETS], dtype=np.int64)
        txout = sess.run(
            None, {'text': text, 'num_buckets': num_buckets})
        try:
            from tensorflow.raw_ops import StringToHashBucket
            dotf = True
        except ImportError:
            dotf = False
        if dotf:
            tfres = StringToHashBucket(
                string_tensor=text, num_buckets=num_buckets[0])
            self.assertEqual(tfres.shape, txout[0].shape)
            self.assertEqual(tfres.numpy().tolist(), txout[0].tolist())
        exp = np.array([[15, 11], [10, 21], [20, 21]], dtype=np.int64)
        self.assertEqual(exp.shape, txout[0].shape)
        self.assertEqual(exp.tolist(), txout[0].tolist())

    def test_string_to_hash_bucket_fast_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_to_hash(
            '', kind='hash_bucket_fast')
        self.assertIn('op_type: "StringToHashBucketFast"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        raw = ["abc", "abcdé", "$$^l!%*ù", "", "a", "A"]
        text = np.array(raw).reshape((3, 2))
        num_buckets = np.array([NUM_BUCKETS], dtype=np.int64)
        txout = sess.run(
            None, {'text': text, 'num_buckets': num_buckets})
        try:
            from tensorflow.raw_ops import StringToHashBucketFast
            dotf = True
        except ImportError:
            dotf = False
        if dotf:
            tfres = StringToHashBucketFast(
                input=text, num_buckets=num_buckets[0])
            self.assertEqual(tfres.shape, txout[0].shape)
            self.assertEqual(tfres.numpy().tolist(), txout[0].tolist())
        exp = np.array([[9, 17], [4, 21], [14, 12]], dtype=np.int64)
        self.assertEqual(exp.shape, txout[0].shape)
        self.assertEqual(exp.tolist(), txout[0].tolist())

    def test_string_to_hash_bucket_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_to_hash(
            'Py', kind='hash_bucket')
        self.assertIn('op_type: "PyStringToHashBucket"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        raw = ["abc", "abcdé", "$$^l!%*ù", "", "a", "A"]
        text = np.array(raw).reshape((3, 2))
        num_buckets = np.array([NUM_BUCKETS], dtype=np.int64)
        exp = np.array([[9, 17], [4, 21], [14, 12]], dtype=np.int64)
        txout = sess.run(
            None, {'text': text, 'num_buckets': num_buckets})
        self.assertEqual(exp.shape, txout[0].shape)
        self.assertEqual(exp.tolist(), txout[0].tolist())

    def enumerate_matrix_couples(self):
        for i in range(1, 5):
            shape = (3,) * i
            a = (np.random.rand(*shape) * 10).astype(np.int32).astype(np.str)
            yield a, a
            for j in range(i):
                shape2 = list(shape)
                shape2[j] = 1
                b = (np.random.rand(*shape2) * 10).astype(
                    np.int32).astype(np.str)
                yield a, b
                for k in range(j+1, i):
                    shape3 = list(shape2)
                    shape3[k] = 1
                    b = (np.random.rand(*shape3) * 10).astype(
                        np.int32).astype(np.str)
                    yield a, b

    def test_string_equal_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_equal('Py')
        self.assertIn('op_type: "PyStringEqual"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        for x, y in self.enumerate_matrix_couples():
            txout = sess.run(None, {'x': x, 'y': y})
            self.assertEqual(txout[0].tolist(), (x == y).tolist())
            txout = sess.run(None, {'x': y, 'y': x})
            self.assertEqual(txout[0].tolist(), (y == x).tolist())

    def test_string_equal_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_equal('')
        self.assertIn('op_type: "StringEqual"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)

        for x, y in self.enumerate_matrix_couples():
            txout = sess.run(None, {'x': x, 'y': y})
            self.assertEqual(txout[0].tolist(), (x == y).tolist())
            txout = sess.run(None, {'x': y, 'y': x})
            self.assertEqual(txout[0].tolist(), (y == x).tolist())

    def test_string_split_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split('Py')
        self.assertIn('op_type: "PyStringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = np.array(["a,,b", "", "aa,b,c", "dddddd"])
        delimiter = np.array([","])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = np.array([skip])

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                if skip_empty:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2], [3, 0]])
                    exp_text = np.array(['a', 'b', 'aa', 'b', 'c', 'dddddd'])
                else:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1],
                         [2, 2], [3, 0]])
                    exp_text = np.array(
                        ['a', '', 'b', 'aa', 'b', 'c', 'dddddd'])
                exp_shape = np.array([4, 3])
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_split_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split('')
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = np.array(["a,,b", "", "aa,b,c", "dddddd"])
        delimiter = np.array([","])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = np.array([skip])

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                try:
                    from tensorflow.raw_ops import StringSplit
                    dotf = True
                except ImportError:
                    dotf = False
                if dotf:
                    tfres = StringSplit(
                        input=input, delimiter=",,", skip_empty=skip)
                    self.assertEqual(
                        [_.decode() for _ in tfres[1].numpy().tolist()],
                        txout[1].tolist())
                    self.assertEqual(
                        tfres[0].numpy().tolist(), txout[0].tolist())
                    self.assertEqual(
                        tfres[2].numpy().tolist(), txout[2].tolist())

                if skip_empty:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [2, 0], [2, 1], [2, 2], [3, 0]])
                    exp_text = np.array(['a', 'b', 'aa', 'b', 'c', 'dddddd'])
                else:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [0, 2], [2, 0], [2, 1],
                         [2, 2], [3, 0]])
                    exp_text = np.array(
                        ['a', '', 'b', 'aa', 'b', 'c', 'dddddd'])
                exp_shape = np.array([4, 3])
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_split_cc_sep2(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split('')
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = np.array(["a*b", "a,*b", "aa,b,,c", 'z', "dddddd,", "**"])
        delimiter = np.array([",*"])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = np.array([skip])

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                try:
                    from tensorflow.raw_ops import StringSplit
                    dotf = True
                except ImportError:
                    dotf = False
                if dotf:
                    tfres = StringSplit(
                        input=input, delimiter=",*", skip_empty=skip)
                    self.assertEqual(
                        [_.decode() for _ in tfres[1].numpy().tolist()],
                        txout[1].tolist())
                    self.assertEqual(
                        tfres[0].numpy().tolist(), txout[0].tolist())
                    self.assertEqual(
                        tfres[2].numpy().tolist(), txout[2].tolist())

                if skip_empty:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1],
                         [2, 2], [3, 0], [4, 0]])
                    exp_text = np.array(
                        ['a', 'b', 'a', 'b', 'aa', 'b', 'c', 'z', 'dddddd'])
                    exp_shape = np.array([6, 3])
                else:
                    exp_indices = np.array(
                        [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [2, 0],
                         [2, 1], [2, 2], [2, 3], [3, 0], [4, 0], [4, 1],
                         [5, 0], [5, 1], [5, 2]])
                    exp_text = np.array(
                        ['a', 'b', 'a', '', 'b', 'aa', 'b', '', 'c',
                         'z', 'dddddd', '', '', '', ''])
                    exp_shape = np.array([6, 4])
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_split_cc_sep0(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_split('')
        self.assertIn('op_type: "StringSplit"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = np.array(["a*b", "a,*b"])
        delimiter = np.array([""])

        for skip in [True, False]:
            with self.subTest(skip=skip):
                skip_empty = np.array([skip])

                txout = sess.run(
                    None, {'input': input, 'delimiter': delimiter,
                           'skip_empty': skip_empty})

                try:
                    from tensorflow.raw_ops import StringSplit
                    dotf = True
                except ImportError:
                    dotf = False
                if dotf:
                    tfres = StringSplit(
                        input=input, delimiter="", skip_empty=skip)
                    self.assertEqual(
                        [_.decode() for _ in tfres[1].numpy().tolist()],
                        txout[1].tolist())
                    self.assertEqual(
                        tfres[0].numpy().tolist(), txout[0].tolist())
                    self.assertEqual(
                        tfres[2].numpy().tolist(), txout[2].tolist())

                exp_indices = np.array(
                    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [1, 3]])
                exp_text = np.array(['a', '*', 'b', 'a', ',', '*', 'b'])
                exp_shape = np.array([2, 4])
                self.assertEqual(exp_text.tolist(), txout[1].tolist())
                self.assertEqual(exp_indices.tolist(), txout[0].tolist())
                self.assertEqual(exp_shape.tolist(), txout[2].tolist())

    def test_string_regex_split_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_regex_split('')
        self.assertIn('op_type: "StringRegexSplitWithOffsets"',
                      str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input = np.array(["hello there", "hello  there"])
        pattern = np.array(["(\\s)"])

        # keep_pattern not empty
        keep_pattern = np.array(["\\s"])
        txout = sess.run(
            None, {'input': input, 'pattern': pattern,
                   'keep_pattern': keep_pattern})

        exp_text = np.array(['hello', ' ', 'there',
                             'hello', ' ', ' ', 'there'])
        exp_begins = np.array([0, 5, 6, 0, 5, 6, 7])
        exp_ends = np.array([5, 6, 11, 5, 6, 7, 12])
        exp_rows = np.array([0, 3, 7])

        self.assertEqual(exp_text.tolist(), txout[0].tolist())
        self.assertEqual(exp_begins.tolist(), txout[1].tolist())
        self.assertEqual(exp_ends.tolist(), txout[2].tolist())
        self.assertEqual(exp_rows.tolist(), txout[3].tolist())

        try:
            from tensorflow_text.python.ops.regex_split_ops import gen_regex_split_ops as lib_gen_regex_split_ops
            use_tf = True
        except ImportError:
            use_tf = False

        if use_tf:
            tf_tokens, tf_begins, tf_ends, tf_rows = lib_gen_regex_split_ops.regex_split_with_offsets(input, "(\\s)", "\\s")
            ltk = [s.decode('utf-8') for s in tf_tokens.numpy()]
            self.assertEqual(ltk, txout[0].tolist())
            self.assertEqual(tf_begins.numpy().tolist(), txout[1].tolist())
            self.assertEqual(tf_ends.numpy().tolist(), txout[2].tolist())
            self.assertEqual(tf_rows.numpy().tolist(), txout[3].tolist())

        # keep_pattern empty
        keep_pattern = np.array([""])
        txout = sess.run(
            None, {'input': input, 'pattern': pattern,
                   'keep_pattern': keep_pattern})
        exp_text = np.array(['hello', 'there', 'hello', 'there'])
        exp_begins = np.array([0, 6, 0, 7])
        exp_ends = np.array([5, 11, 5, 12])
        exp_rows = np.array([0, 2, 4])

        self.assertEqual(exp_text.tolist(), txout[0].tolist())
        self.assertEqual(exp_begins.tolist(), txout[1].tolist())
        self.assertEqual(exp_ends.tolist(), txout[2].tolist())
        self.assertEqual(exp_rows.tolist(), txout[3].tolist())

        if use_tf:
            tf_tokens, tf_begins, tf_ends, tf_rows = lib_gen_regex_split_ops.regex_split_with_offsets(input, "(\\s)", "")
            ltk = [s.decode('utf-8') for s in tf_tokens.numpy()]
            self.assertEqual(ltk, txout[0].tolist())
            self.assertEqual(tf_begins.numpy().tolist(), txout[1].tolist())
            self.assertEqual(tf_ends.numpy().tolist(), txout[2].tolist())
            self.assertEqual(tf_rows.numpy().tolist(), txout[3].tolist())

    def test_string_wordpiece_tokenizer_cc(self):
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
               np.array([0, 5, 11], dtype=np.int64),
               np.array([0, 5], dtype=np.int32),
               np.array([5, 11], dtype=np.int32),
               np.array(['unwanted', 'running', 'unwantedX', 'running']),
               np.array([0, 2, 4], dtype=np.int64)]

        def check(o1, o2):
            try:
                assert_almost_equal(o1, o2)
            except TypeError:
                assert o1.tolist() == o2.tolist()

        try:
            from tensorflow_text.python.ops.wordpiece_tokenizer import gen_wordpiece_tokenizer as lib_gen
            import tensorflow as tf
            use_tf = True
        except ImportError:
            use_tf = False

        if use_tf:
            def _CreateTable(vocab, num_oov=1):
                init = tf.lookup.KeyValueTensorInitializer(
                    vocab,
                    tf.range(tf.size(vocab, out_type=tf.int64), dtype=tf.int64),
                    key_dtype=tf.string,
                    value_dtype=tf.int64)
                res = tf.lookup.StaticVocabularyTable(init, num_oov, lookup_key_dtype=tf.string)
                res.__len__ = lambda self: len(vocab)
                  

            vocab_table = _CreateTable(["want", "##want", "##ed", "wa", "un", "runn", "##ing"])

            text = tf.convert_to_tensor(["unwanted running", "unwantedX running"], dtype=tf.string)
            try:
                tf_tokens, tf_rows, tf_begins, tf_ends = (
                          lib_gen.wordpiece_tokenize_with_offsets(
                              input_values=text,
                              vocab_lookup_table=vocab_table,
                              suffix_indicator="##",
                              use_unknown_token=True,
                              max_bytes_per_word=100,
                              max_chars_per_token=0,
                              unknown_token="[UNK]",
                              split_unknown_characters=False))

                ltk = [s.decode('utf-8') for s in tf_tokens.numpy()]
                check(ltk, txout[0])
                check(tf_rows.numpy(), txout[1])
                check(tf_begins.numpy(), txout[2])
                check(tf_ends.numpy(), txout[3])
            except ValueError:
                # Issue here.
                pass

        check(exp[0], cc_txout[0])
        check(exp[1], cc_txout[1])
        check(exp[2], cc_txout[2])
        check(exp[3], cc_txout[3])
        check(exp[4], cc_txout[4])
        check(exp[5], cc_txout[5])


if __name__ == "__main__":
    unittest.main()

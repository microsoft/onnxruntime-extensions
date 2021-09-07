# coding: utf-8
import unittest
import re
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_extensions import (make_onnx_model,
                                    get_library_path as _get_library_path)


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
                '%sStringECMARegexReplace' % prefix, ['id1', 'id2', 'id3'],
                ['customout'], domain=domain))
    else:
        nodes.append(
            helper.make_node(
                '%sStringECMARegexReplace' % prefix, ['id1', 'id2', 'id3'],
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
    model = make_onnx_model(graph)
    return model


def _create_test_model_string_regex_split(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(helper.make_node('Identity', ['input'], ['id1']))
    nodes.append(helper.make_node('Identity', ['pattern'], ['id2']))
    nodes.append(helper.make_node('Identity', ['keep_pattern'], ['id3']))
    nodes.append(
        helper.make_node(
            '%sStringECMARegexSplitWithOffsets' % prefix, ['id1', 'id2', 'id3'],
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
    model = make_onnx_model(graph)
    return model


class TestStringECMARegex(unittest.TestCase):
    def test_string_replace_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('')
        self.assertIn('op_type: "StringECMARegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_$1(void) {'])
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
        self.assertIn('op_type: "StringECMARegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_$1(void) {'])
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
        self.assertIn('op_type: "StringECMARegexReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_$1(void) {'])
        text = np.array([['def myfunc():'], ['def dummy():' * 2]])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject* py_myfunc(void) {'],
               ['static PyObject* py_dummy(void) {' * 2]]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_regex_split_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_regex_split('')
        self.assertIn('op_type: "StringECMARegexSplitWithOffsets"',
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
            tf_tokens, tf_begins, tf_ends, tf_rows = lib_gen_regex_split_ops.regex_split_with_offsets(input, "(\\s)",
                                                                                                      "\\s")
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
            tf_tokens, tf_begins, tf_ends, tf_rows = lib_gen_regex_split_ops.regex_split_with_offsets(input, "(\\s)",
                                                                                                      "")
            ltk = [s.decode('utf-8') for s in tf_tokens.numpy()]
            self.assertEqual(ltk, txout[0].tolist())
            self.assertEqual(tf_begins.numpy().tolist(), txout[1].tolist())
            self.assertEqual(tf_ends.numpy().tolist(), txout[2].tolist())
            self.assertEqual(tf_rows.numpy().tolist(), txout[3].tolist())

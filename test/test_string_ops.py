# coding: utf-8
import unittest
import re
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from ortcustomops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_string_upper(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('%sStringUpper' % prefix,
                                  ['identity1'], ['customout'],
                                  domain=domain)]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None, 1])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None, 1])

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
        helper.make_node(
            '%sStringJoin' % prefix, ['identity1', 'identity2'],
            ['customout'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'text', onnx_proto.TensorProto.STRING, [None, None])
    input1 = helper.make_tensor_value_info(
        'sep', onnx_proto.TensorProto.STRING, [1])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.STRING, [None, 1])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_string_replace(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(
        helper.make_node('Identity', ['text'], ['id1']))
    nodes.append(
        helper.make_node('Identity', ['pattern'], ['id2']))
    nodes.append(
        helper.make_node('Identity', ['rewrite'], ['id3']))
    nodes.append(
        helper.make_node(
            '%sStringReplace' % prefix, ['id1', 'id2', 'id3'],
            ['customout'], domain=domain))

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


class TestPythonOpString(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="PyStringUpper",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_upper(x):
            # The user custom op implementation here.
            return np.array([s.upper() for s in x.ravel()]).reshape(x.shape)

        @onnx_op(op_type="PyStringJoin",
                 inputs=[PyCustomOpDef.dt_string, PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string])
        def string_join(x, sep):
            # The user custom op implementation here.
            if sep.shape != (1, ):
                raise RuntimeError(
                    "Unexpected shape {} for 'sep'.".format(sep.shape))
            sp = sep[0]
            return np.array([sp.join(row) for row in x])

        @onnx_op(op_type="PyStringReplace",
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

    def test_string_upper_cc_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abcé"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABCé"]]).tolist())

    def test_string_upper_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('Py')
        self.assertIn('op_type: "PyStringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABC"]]).tolist())

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
        txout = sess.run(None, {'text': text, 'sep': sep})
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;c", "aa;bb;"]).tolist())

    def test_string_join_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_join('')
        self.assertIn('op_type: "StringJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        text = np.vstack([np.array([["a", "b", "c"]]),
                          np.array([["aa", "bb", ""]])])
        sep = np.array([";"])
        txout = sess.run(None, {'text': text, 'sep': sep})
        self.assertEqual(
            txout[0].tolist(), np.array(["a;b;c", "aa;bb;"]).tolist())

    def test_string_replace_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('')
        self.assertIn('op_type: "StringReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject* py_\1(void) {'])
        text = np.array([['def myfunc():'], ['def dummy():']])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject* py_myfunc(void) {'],
               ['static PyObject* py_dummy(void) {']]
        self.assertEqual(exp, txout[0].tolist())

    def test_string_replace_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_replace('Py')
        self.assertIn('op_type: "PyStringReplace"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        pattern = np.array([r'def\s+([a-zA-Z_][a-zA-Z_0-9]*)\s*\(\s*\):'])
        rewrite = np.array([r'static PyObject*\npy_\1(void)\n{'])
        text = np.array([['def myfunc():'], ['def dummy():']])
        txout = sess.run(
            None, {'text': text, 'pattern': pattern, 'rewrite': rewrite})
        exp = [['static PyObject*\npy_myfunc(void)\n{'],
               ['static PyObject*\npy_dummy(void)\n{']]
        self.assertEqual(exp, txout[0].tolist())


if __name__ == "__main__":
    unittest.main()

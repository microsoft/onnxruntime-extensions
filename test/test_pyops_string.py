# coding: utf-8
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from ortcustomops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_string_upper(domain):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('StringUpper',
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


@onnx_op(op_type="StringUpper",
         inputs=[PyCustomOpDef.dt_string],
         outputs=[PyCustomOpDef.dt_string])
def string_upper(x):
    # The user custom op implementation here.
    return np.array([s.upper() for s in x.ravel()]).reshape(x.shape)


class TestPythonOpString(unittest.TestCase):

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
        onnx_model = _create_test_model_string_upper('test.customop')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABC"]]).tolist())

    def test_string_upper_cc_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('test.customop')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abcé"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABCé"]]).tolist())

    def test_string_upper_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('ai.onnx.contrib')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abc"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(), np.array([["ABC"]]).tolist())

    def test_string_upper_python_accent(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_string_upper('ai.onnx.contrib')
        self.assertIn('op_type: "StringUpper"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([["Abcé"]])
        txout = sess.run(None, {'input_1': input_1})
        self.assertEqual(txout[0].tolist(),
                         np.array([["ABCé".upper()]]).tolist())


if __name__ == "__main__":
    unittest.main()

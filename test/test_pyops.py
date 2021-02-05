import os
import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_test():
    nodes = []
    nodes.append(helper.make_node(
        'CustomOpOne', ['input_1', 'input_2'], ['output_1'],
        domain='ai.onnx.contrib'))
    nodes.append(helper.make_node(
        'CustomOpTwo', ['output_1'], ['output'],
        domain='ai.onnx.contrib'))

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [3, 5])
    input1 = helper.make_tensor_value_info(
        'input_2', onnx_proto.TensorProto.FLOAT, [3, 5])
    output0 = helper.make_tensor_value_info(
        'output', onnx_proto.TensorProto.INT32, [3, 5])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)])
    return model


def _create_test_model():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('PyReverseMatrix',
                                  ['identity1'], ['reversed'],
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
    output0 = helper.make_tensor_value_info(
        'reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)])
    return model


def _create_test_model_double(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('%sAddEpsilon' % prefix,
                                  ['identity1'], ['customout'],
                                  domain=domain)]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.DOUBLE, [None, None])
    output0 = helper.make_tensor_value_info(
        'customout', onnx_proto.TensorProto.DOUBLE, [None, None])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_model_2outputs(prefix, domain='ai.onnx.contrib'):
    nodes = [
        helper.make_node('Identity', ['x'], ['identity1']),
        helper.make_node(
            '%sNegPos' % prefix, ['identity1'], ['neg', 'pos'],
            domain=domain)
    ]

    input0 = helper.make_tensor_value_info(
        'x', onnx_proto.TensorProto.FLOAT, [])
    output1 = helper.make_tensor_value_info(
        'neg', onnx_proto.TensorProto.FLOAT, [])
    output2 = helper.make_tensor_value_info(
        'pos', onnx_proto.TensorProto.FLOAT, [])

    graph = helper.make_graph(nodes, 'test0', [input0], [output1, output2])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


def _create_test_join():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('PyOpJoin',
                                  ['identity1'], ['joined'],
                                  sep=';',
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.STRING, [None, None])
    output0 = helper.make_tensor_value_info(
        'joined', onnx_proto.TensorProto.STRING, [None])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)])
    return model


class TestPythonOp(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="CustomOpOne",
                 inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float])
        def custom_one_op(x, y):
            return np.add(x, y)

        @onnx_op(op_type="CustomOpTwo",
                 outputs=[PyCustomOpDef.dt_int32])
        def custom_two_op(f):
            return np.round(f).astype(np.int32)

        @onnx_op(op_type="PyReverseMatrix")
        def reverse_matrix(x):
            # The user custom op implementation here.
            return np.flip(x, axis=0).astype(np.float32)

        @onnx_op(op_type="PyAddEpsilon",
                 inputs=[PyCustomOpDef.dt_double],
                 outputs=[PyCustomOpDef.dt_double])
        def add_epsilon(x):
            # The user custom op implementation here.
            return x + 1e-3

        @onnx_op(op_type="PyNegPos",
                 inputs=[PyCustomOpDef.dt_float],
                 outputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_float])
        def negpos(x):
            neg = x.copy()
            pos = x.copy()
            neg[x > 0] = 0
            pos[x < 0] = 0
            return neg, pos

        @onnx_op(op_type="PyOpJoin",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_string],
                 attrs=['sep'])
        def join(xs, **kwargs):
            sep = kwargs.get('sep', '')
            res = []
            for x in xs:
                res.append(sep.join(x))
            return np.array(res, dtype=np.object)

    def test_python_operator(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model()
        self.assertIn('op_type: "PyReverseMatrix"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array(
            [1, 2, 3, 4, 5, 6]).astype(np.float32).reshape([3, 2])
        txout = sess.run(None, {'input_1': input_1})
        assert_almost_equal(txout[0], np.array([[5., 6.], [3., 4.], [1., 2.]]))

    def test_add_epsilon_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_double('Py')
        self.assertIn('op_type: "PyAddEpsilon"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([[0., 1., 1.5], [7., 8., -5.5]])
        txout = sess.run(None, {'input_1': input_1})
        diff = txout[0] - input_1 - 1e-3
        assert_almost_equal(diff, np.zeros(diff.shape))

    def test_python_negpos(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_2outputs('Py')
        self.assertIn('op_type: "PyNegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        x = np.array([[0., 1., 1.5], [7., 8., -5.5]]).astype(np.float32)
        neg, pos = sess.run(None, {'x': x})
        diff = x - (neg + pos)
        assert_almost_equal(diff, np.zeros(diff.shape))

    def test_cc_negpos(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_2outputs("")
        self.assertIn('op_type: "NegPos"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        x = np.array([[0., 1., 1.5], [7., 8., -5.5]]).astype(np.float32)
        neg, pos = sess.run(None, {'x': x})
        diff = x - (neg + pos)
        assert_almost_equal(diff, np.zeros(diff.shape))

    def test_check_saved_model(self):
        this = os.path.dirname(__file__)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_content = _create_test_model_test()
        onnx_bytes = onnx_content.SerializeToString()
        with open(os.path.join(this, 'data', 'custom_op_test.onnx'),
                  'rb') as f:
            saved = f.read()
        assert onnx_bytes == saved

    def test_cc_operator(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_content = _create_test_model_test()
        self.assertIn('op_type: "CustomOpOne"', str(onnx_content))
        ser = onnx_content.SerializeToString()
        sess0 = _ort.InferenceSession(ser, so)
        res = sess0.run(None, {
            'input_1': np.random.rand(3, 5).astype(np.float32),
            'input_2': np.random.rand(3, 5).astype(np.float32)})
        self.assertEqual(res[0].shape, (3, 5))

    def test_python_join(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_join()
        self.assertIn('op_type: "PyOpJoin"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        arr = np.array([["a", "b"]], dtype=np.object)
        txout = sess.run(None, {'input_1': arr})
        exp = np.array(["a;b"], dtype=np.object)
        assert txout[0][0] == exp[0]


if __name__ == "__main__":
    unittest.main()

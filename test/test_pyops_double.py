import unittest
import numpy as np
from numpy.testing import assert_almost_equal
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from ortcustomops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_double(domain):
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('AddEpsilon',
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


@onnx_op(op_type="AddEpsilon",
         inputs=[PyCustomOpDef.dt_double],
         outputs=[PyCustomOpDef.dt_double])
def add_epsilon(x):
    # The user custom op implementation here.
    return x + 1e-3


class TestPythonOpDouble(unittest.TestCase):

    def test_add_epsilon_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_double('ai.onnx.contrib')
        self.assertIn('op_type: "AddEpsilon"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        input_1 = np.array([[0., 1., 1.5], [7., 8., -5.5]])
        txout = sess.run(None, {'input_1': input_1})
        diff = txout[0] - input_1 - 1e-3
        assert_almost_equal(diff, np.zeros(diff.shape))


if __name__ == "__main__":
    unittest.main()

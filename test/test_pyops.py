import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from ortcustomops import (
    onnx_op,
    get_library_path as _get_library_path)


def _create_test_model():
    nodes = []
    nodes[0:] = [helper.make_node('Identity', ['input_1'], ['identity1'])]
    nodes[1:] = [helper.make_node('ReverseMatrix',
                                  ['identity1'], ['reversed'],
                                  domain='ai.onnx.contrib')]

    input0 = helper.make_tensor_value_info('input_1', onnx_proto.TensorProto.FLOAT, [None, 2])
    output0 = helper.make_tensor_value_info('reversed', onnx_proto.TensorProto.FLOAT, [None, 2])

    graph = helper.make_graph(nodes, 'test0', [input0], [output0])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('ai.onnx.contrib', 1)])
    return model


@onnx_op(op_type="ReverseMatrix")
def reverse_matrix(x):
    # the user custom op implementation here:
    return np.flip(x, axis=0)


# TODO: refactor the following code into pytest cases, right now, the script is more friendly for debugging.
so = _ort.SessionOptions()
so.register_custom_ops_library(_get_library_path())

sess0 = _ort.InferenceSession('./test/data/custom_op_test.onnx', so)

res = sess0.run(None, {
    'input_1': np.random.rand(3, 5).astype(np.float32), 'input_2': np.random.rand(3, 5).astype(np.float32)})

print(res[0])

sess = _ort.InferenceSession(_create_test_model().SerializeToString(), so)

txout = sess.run(None, {
    'input_1': np.array([1, 2, 3, 4, 5, 6]).astype(np.float32).reshape([3, 2])})

print(txout[0])

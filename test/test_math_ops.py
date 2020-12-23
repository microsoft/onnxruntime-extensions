# coding: utf-8
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


def _create_test_model_segment_sum(prefix, domain='ai.onnx.contrib'):
    nodes = []
    nodes.append(helper.make_node('Identity', ['data'], ['id1']))
    nodes.append(helper.make_node('Identity', ['segment_ids'], ['id2']))
    nodes.append(
        helper.make_node(
            '%sSegmentSum' % prefix, ['id1', 'id2'], ['z'], domain=domain))

    input0 = helper.make_tensor_value_info(
        'data', onnx_proto.TensorProto.FLOAT, [])
    input1 = helper.make_tensor_value_info(
        'segment_ids', onnx_proto.TensorProto.INT64, [])
    output0 = helper.make_tensor_value_info(
        'z', onnx_proto.TensorProto.FLOAT, [])

    graph = helper.make_graph(nodes, 'test0', [input0, input1], [output0])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid(domain, 1)])
    return model


class TestMathOpString(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

        @onnx_op(op_type="PySegmentSum",
                 inputs=[PyCustomOpDef.dt_float, PyCustomOpDef.dt_int64],
                 outputs=[PyCustomOpDef.dt_float])
        def segment_sum(data, segment_ids):
            # segment_ids is sorted
            nb_seg = segment_ids[-1] + 1
            sh = (nb_seg, ) + data.shape[1:]
            res = np.zeros(sh, dtype=data.dtype)
            for seg, row in zip(segment_ids, data):
                res[seg] += row
            return res

    def test_segment_sum_cc(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_segment_sum('')
        self.assertIn('op_type: "SegmentSum"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        data = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]],
                        dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int64)
        exp = np.array([[5, 5, 5, 5], [5, 6, 7, 8]], dtype=np.float32)
        txout = sess.run(None, {'data': data, 'segment_ids': segment_ids})
        self.assertEqual(exp.shape, txout[0].shape)
        self.assertEqual(exp.tolist(), txout[0].tolist())

    def test_segment_sum_python(self):
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        onnx_model = _create_test_model_segment_sum('Py')
        self.assertIn('op_type: "PySegmentSum"', str(onnx_model))
        sess = _ort.InferenceSession(onnx_model.SerializeToString(), so)
        data = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [5, 6, 7, 8]],
                        dtype=np.float32)
        segment_ids = np.array([0, 0, 1], dtype=np.int64)
        exp = np.array([[5, 5, 5, 5], [5, 6, 7, 8]], dtype=np.float32)
        txout = sess.run(None, {'data': data, 'segment_ids': segment_ids})
        self.assertEqual(exp.shape, txout[0].shape)
        self.assertEqual(exp.tolist(), txout[0].tolist())

        try:
            from tensorflow.raw_ops import SegmentSum
            dotf = True
        except ImportError:
            dotf = False
        if dotf:
            tfres = SegmentSum(data=data, segment_ids=segment_ids)
            self.assertEqual(tfres.shape, txout[0].shape)
            self.assertEqual(tfres.numpy().tolist(), txout[0].tolist())


if __name__ == "__main__":
    unittest.main()

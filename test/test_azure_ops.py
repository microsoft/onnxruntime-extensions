import unittest
import numpy as np
import numpy.lib.recfunctions as nlr
import onnxruntime as _ort
from pathlib import Path
from distutils.version import LooseVersion
from onnx import helper, TensorProto, onnx_pb as onnx_proto
from onnxruntime_extensions import (
    make_onnx_model,
    get_library_path as _get_library_path,
    PyOrtFunction)

class TestAzureOps(unittest.TestCase):

    @classmethod
    def createTestModel(cls):
        auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [-1])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [-1])
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [-1])
        invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                                   domain='ai.onnx.contrib', name='triton_invoker',
                                   model_uri='https://endpoint-8344333.westus2.inference.ml.azure.com',
                                   model_name='addf',model_version='1', verbose='1')
        graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
        model = helper.make_model(graph)
        return model

    @classmethod
    def setUpClass(cls) -> None:
        pass

    def test_triton_invoker(self):
        if LooseVersion(_ort.__version__) >= LooseVersion("1.15.1"):
            onnx_model = TestAzureOps.createTestModel()
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
            auth_token = np.array(['XxvN8QTfK7AwzOyHSenaBUlvzGgdSEc4'])
            x = np.array([1,2,3,4]).astype(np.float32)
            y = np.array([4,3,2,1]).astype(np.float32)
            ort_inputs = {
                "auth_token": auth_token,
                "X": x,
                "Y": y
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertEqual((out == np.array([5,5,5,5]).astype(np.float32)).all(), True)

if __name__ == "__main__":
    unittest.main()
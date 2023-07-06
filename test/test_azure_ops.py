import os
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
    def createAzureTritonModel(cls):
        auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
        X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [-1])
        Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [-1])
        Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [-1])
        invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                                   domain='ai.onnx.contrib', name='triton_invoker',
                                   model_uri='...',
                                   model_name='addf', model_version='1', verbose='1')
        graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
        model = helper.make_model(graph)
        return model

    @classmethod
    def createAzureAudioModel(cls):
        auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
        X = helper.make_tensor_value_info('X', TensorProto.UINT8, [-1])
        Y = helper.make_tensor_value_info('Y', TensorProto.STRING, [-1])
        invoker = helper.make_node('AzureAudioInvoker', ['auth_token', 'X'], ['Y'],
                                   domain='ai.onnx.contrib', name='audio_invoker',
                                   model_uri='...',
                                   model_name='whisper', model_version='1', verbose='1')
        graph = helper.make_graph([invoker], 'graph', [auth_token, X], [Y])
        model = helper.make_model(graph)
        return model

    @classmethod
    def isEnabled(cls):
        config = "TEST_AZURE_INVOKERS_AS_CPU_OP"
        return config in os.environ and os.environ[config] == 'ON'

    def test_azure_triton_invoker(self):
        if TestAzureOps.isEnabled():
            if LooseVersion(_ort.__version__) >= LooseVersion("1.15.1"):
                onnx_model = TestAzureOps.createAzureTritonModel()
                so = _ort.SessionOptions()
                so.register_custom_ops_library(_get_library_path())
                _ = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
                print ("azure triton invoker session loaded.")
        else:
            print ("test_azure_triton_invoker disabled.")

    def test_azure_audio_invoker(self):
        if TestAzureOps.isEnabled():
            onnx_model = TestAzureOps.createAzureAudioModel()
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            _ = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=['CPUExecutionProvider'])
            print ("azure audio invoker session loaded.")
        else:
            print ("test_azure_audio_invoker disabled.")

if __name__ == "__main__":
    unittest.main()
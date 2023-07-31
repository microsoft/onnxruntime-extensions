# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import unittest
import numpy as np

from onnx import checker, helper, onnx_pb as onnx_proto
from onnxruntime_extensions import PyOrtFunction, util, get_library_path
from onnxruntime import *

script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, ".."))
test_data_dir = os.path.join(ort_ext_root, "test", "data", "azure")

class TestAzureOps(unittest.TestCase):

    def test_addf(self):
        opt = SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        sess = InferenceSession(os.path.join(test_data_dir, "triton_addf.onnx"),
                                opt, providers=["CPUExecutionProvider"])
        auth_token = np.array(['RuB4VFQMfRlOPadcdnuTjYxQi25bwSHa'])
        x = np.array([1,2,3,4]).astype(np.float32)
        y = np.array([4,3,2,1]).astype(np.float32)
        ort_inputs = {
            "auth_token": auth_token,
            "X": x,
            "Y": y
        }
        out = sess.run(None, ort_inputs)[0]
        self.assertTrue(np.allclose(out, [5,5,5,5]))

    def testAddf8(self):
        opt = SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        sess = InferenceSession(os.path.join(test_data_dir, "triton_addf8.onnx"),
                                opt, providers=["CPUExecutionProvider"])
        auth_token = np.array(['cBdnbqR22qYWXR8ImY7tPGU2WYggGfPz'])
        x = np.array([1,2,3,4]).astype(np.double)
        y = np.array([4,3,2,1]).astype(np.double)
        ort_inputs = {
            "auth_token": auth_token,
            "X": x,
            "Y": y
        }
        out = sess.run(None, ort_inputs)[0]
        self.assertTrue(np.allclose(out, [5,5,5,5]))

    def testAddi4(self):
        opt = SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        sess = InferenceSession(os.path.join(test_data_dir, "triton_addi4.onnx"),
                                opt, providers=["CPUExecutionProvider"])
        auth_token = np.array(['MYFF7rX0Dxr0oVzK5Dac77RPZL6TnrNf'])
        x = np.array([1,2,3,4]).astype(np.int32)
        y = np.array([4,3,2,1]).astype(np.int32)
        ort_inputs = {
            "auth_token": auth_token,
            "X": x,
            "Y": y
        }
        out = sess.run(None, ort_inputs)[0]
        self.assertTrue(np.allclose(out, [5,5,5,5]))

    def testAnd(self):
        opt = SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        sess = InferenceSession(os.path.join(test_data_dir, "triton_and.onnx"),
                                opt, providers=["CPUExecutionProvider"])
        auth_token = np.array(['QmzQoWY9GzUH6yihHaLd43g3RScXppjg'])
        x = np.array([True, True])
        y = np.array([True, False])
        ort_inputs = {
            "auth_token": auth_token,
            "X": x,
            "Y": y
        }
        out = sess.run(None, ort_inputs)[0]
        self.assertTrue(np.allclose(out, [True, False]))

    def testStr(self):
        opt = SessionOptions()
        opt.register_custom_ops_library(get_library_path())
        sess = InferenceSession(os.path.join(test_data_dir, "triton_str.onnx"),
                                opt, providers=["CPUExecutionProvider"])
        auth_token = np.array(['dyd4Ml9hvZX3cF3ticHyQ7ZD7RVVfaZH'])
        str_in = np.array(['this is the input'])
        ort_inputs = {
            "auth_token": auth_token,
            "str_in": str_in
        }
        outs = sess.run(None, ort_inputs)
        self.assertEqual(len(outs), 2)
        self.assertEqual(outs[0], ['this is the input'])
        self.assertEqual(outs[1], ['this is the input'])


if __name__ == '__main__':
    unittest.main()
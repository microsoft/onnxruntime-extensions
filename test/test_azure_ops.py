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

    def __init__(self, config):
        super().__init__(config)
        self.__enabled = os.getenv('OCOS_ENABLE_AZURE','') == '1'
        if self.__enabled:
            self.__opt = SessionOptions()
            self.__opt.register_custom_ops_library(get_library_path())

    def test_add_f(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addf.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('ADDF', '')])
            x = np.array([1,2,3,4]).astype(np.float32)
            y = np.array([4,3,2,1]).astype(np.float32)
            ort_inputs = {
                "auth_token": auth_token,
                "X": x,
                "Y": y
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue(np.allclose(out, [5,5,5,5]))

    def test_add_f8(self):
        if self.__enabled:
            opt = SessionOptions()
            opt.register_custom_ops_library(get_library_path())
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addf8.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('ADDF8', '')])
            x = np.array([1,2,3,4]).astype(np.double)
            y = np.array([4,3,2,1]).astype(np.double)
            ort_inputs = {
                "auth_token": auth_token,
                "X": x,
                "Y": y
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue(np.allclose(out, [5,5,5,5]))

    def test_add_i4(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addi4.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('ADDI4', '')])
            x = np.array([1,2,3,4]).astype(np.int32)
            y = np.array([4,3,2,1]).astype(np.int32)
            ort_inputs = {
                "auth_token": auth_token,
                "X": x,
                "Y": y
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue(np.allclose(out, [5,5,5,5]))

    def test_and(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_and.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('AND', '')])
            x = np.array([True, True])
            y = np.array([True, False])
            ort_inputs = {
                "auth_token": auth_token,
                "X": x,
                "Y": y
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue(np.allclose(out, [True, False]))

    def test_str(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_str.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('STR', '')])
            str_in = np.array(['this is the input'])
            ort_inputs = {
                "auth_token": auth_token,
                "str_in": str_in
            }
            outs = sess.run(None, ort_inputs)
            self.assertEqual(len(outs), 2)
            self.assertEqual(outs[0], ['this is the input'])
            self.assertEqual(outs[1], ['this is the input'])

    def test_open_ai_audio(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "openai_audio.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('AUDIO', '')])
            model = np.array(['whisper-1'])
            response_format = np.array(['text'])

            with open(os.path.join(test_data_dir, "test16.wav"), "rb") as _f:
                audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
                ort_inputs = {
                    "auth_token": auth_token,
                    "model_name": model,
                    "response_format": response_format,
                    "file": audio_blob,
                }
                out = sess.run(None, ort_inputs)[0]
                self.assertEqual(out, ['This is a test recording to test the Whisper model.\n'])

    def test_azure_chat(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "azure_chat.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider", "AzureExecutionProvider"])
            auth_token = np.array([os.getenv('CHAT', '')])
            chat = np.array([r'{"messages":[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},{"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},{"role": "user", "content": "Do other Azure AI services support this too?"}]}'])
            ort_inputs = {
                "auth_token": auth_token,
                "chat": chat,
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue('chat.completion' in out[0])


if __name__ == '__main__':
    unittest.main()
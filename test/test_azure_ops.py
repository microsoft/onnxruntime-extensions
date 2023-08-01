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
        self.__enabled = os.environ['OCOS_ENABLE_AZURE'] == '1'
        if self.__enabled:
            self.__opt = SessionOptions()
            self.__opt.register_custom_ops_library(get_library_path())

    def test_addf(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addf.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['ADDF']])
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
        if self.__enabled:
            opt = SessionOptions()
            opt.register_custom_ops_library(get_library_path())
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addf8.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['ADDF8']])
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
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_addi4.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['ADDI4']])
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
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_and.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['AND']])
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
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "triton_str.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['STR']])
            str_in = np.array(['this is the input'])
            ort_inputs = {
                "auth_token": auth_token,
                "str_in": str_in
            }
            outs = sess.run(None, ort_inputs)
            self.assertEqual(len(outs), 2)
            self.assertEqual(outs[0], ['this is the input'])
            self.assertEqual(outs[1], ['this is the input'])

    def testOpenAiAudio(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "openai_audio.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['AUDIO']])
            model = np.array(['whisper-1'])
            response_format = np.array(['text'])

            with open(os.path.join(test_data_dir, "test16.wav"), "rb") as _f:
                audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
                ort_inputs = {
                    "auth_token": auth_token,
                    "model": model,
                    "response_format": response_format,
                    "file": audio_blob
                }
                out = sess.run(None, ort_inputs)[0]
                self.assertEqual(out, ['This is a test recording to test the Whisper model.\n'])

    def testOpenAiChat(self):
        if self.__enabled:
            sess = InferenceSession(os.path.join(test_data_dir, "openai_chat.onnx"),
                                    self.__opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['CHAT']])
            chat = np.array(['{\"model\": \"gpt-3.5-turbo\",\"messages\": [{\"role\": \"system\", \"content\": \"You are a helpful assistant.\"}, {\"role\": \"user\", \"content\": \"Hello!\"}]}'])
            ort_inputs = {
                "auth_token": auth_token,
                "chat": chat,
            }
            out = sess.run(None, ort_inputs)[0]
            self.assertTrue('assist' in out[0])

    def testOpenAiEmb(self):
        if self.__enabled:
            opt = SessionOptions()
            opt.register_custom_ops_library(get_library_path())
            sess = InferenceSession(os.path.join(test_data_dir, "openai_embedding.onnx"),
                                    opt, providers=["CPUExecutionProvider"])
            auth_token = np.array([os.environ['EMB']])
            text = np.array(['{\"input\": \"The food was delicious and the waiter...\", \"model\": \"text-embedding-ada-002\"}'])

            ort_inputs = {
                "auth_token": auth_token,
                "text": text,
            }

            out = sess.run(None, ort_inputs)[0]
            self.assertTrue('text-embedding-ada' in out[0])


if __name__ == '__main__':
    unittest.main()
import unittest
import numpy as np
import onnxruntime as _ort

from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from transformers import GPT2Tokenizer
from onnxruntime_customops import (
    onnx_op,
    enable_custom_op,
    PyCustomOpDef,
    expand_onnx_inputs,
    get_library_path as _get_library_path)


def _get_file_content(path):
    with open(path, "rb") as file:
        return file.read()


def _get_test_data_file(*sub_dirs):
    test_dir = Path(__file__).parent
    return str(test_dir.joinpath(*sub_dirs))


def _create_test_model():
    nodes = [helper.make_node(
        'Identity', ['input_1'], ['output_1'])]

    input1 = helper.make_tensor_value_info(
        'input_1', onnx_proto.TensorProto.INT64, [None, None])
    output1 = helper.make_tensor_value_info(
        'output_1', onnx_proto.TensorProto.INT64, [None, None])

    graph = helper.make_graph(nodes, 'test0', [input1], [output1])
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid('', 12)])
    return model


def _bind_tokenizer(model, **kwargs):
    vocab_file = kwargs["vocab_file"]
    merges_file = kwargs["merges_file"]

    return expand_onnx_inputs(
        model, 'input_1',
        [helper.make_node(
            'GPT2Tokenizer', ['string_input'], ['input_1'],
            vocab=_get_file_content(vocab_file),
            merges=_get_file_content(merges_file), name='bpetok',
            domain='ai.onnx.contrib')],
        [helper.make_tensor_value_info(
            'string_input', onnx_proto.TensorProto.STRING, [None])],
    )


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokjson = _get_test_data_file('data', 'gpt2.vocab')
        merges = tokjson.replace('.vocab', '.merges.txt')
        cls.tokenizer = GPT2Tokenizer(tokjson, merges)

        model = _create_test_model()
        cls.binded_model = _bind_tokenizer(
            model, vocab_file=tokjson, merges_file=merges)

        @onnx_op(op_type="GPT2Tokenizer",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_int64])
        def bpe_toenizer(s):
            # The user custom op implementation here.
            TestGPT2Tokenizer.pyop_invoked = True
            return np.array(
                [TestGPT2Tokenizer.tokenizer.encode(st_) for st_ in s])

    def _run_tokenizer(self, pyop_flag, test_sentence):
        enable_custom_op(pyop_flag)

        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(self.binded_model.SerializeToString(), so)
        input_text = np.array([test_sentence])
        txtout = sess.run(None, {'string_input': input_text})

        np.testing.assert_array_equal(
            txtout[0], np.array([self.tokenizer.encode(test_sentence)]))
        del sess
        del so

    def test_tokenizer(self):
        TestGPT2Tokenizer.pyop_invoked = False

        self._run_tokenizer(False, "I can feel the magic, can you?")
        self.assertFalse(TestGPT2Tokenizer.pyop_invoked)

        self._run_tokenizer(False, "Hey Cortana")
        self._run_tokenizer(False, "你好123。david")
        self._run_tokenizer(False, "women'thinsulate 3 button leather car co")
        self._run_tokenizer(False, "#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ")
        self._run_tokenizer(False, "ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ")
        self._run_tokenizer(False, "⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏")

    def test_tokenizer_pyop(self):
        TestGPT2Tokenizer.pyop_invoked = False
        self._run_tokenizer(True, "I can feel the magic, can you?")
        self.assertTrue(TestGPT2Tokenizer.pyop_invoked)

        self._run_tokenizer(True, "Hey Cortana")
        self._run_tokenizer(True, "你好123。david")
        self._run_tokenizer(True, "women'thinsulate 3 button leather car co")
        self._run_tokenizer(True, "#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ")
        self._run_tokenizer(True, "ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ")
        self._run_tokenizer(True, "⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏")


if __name__ == "__main__":
    unittest.main()

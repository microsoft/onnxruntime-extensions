import unittest
import numpy as np
import onnxruntime as _ort

from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from transformers import *
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


def _create_test_model(**kwargs):
    vocab_file = kwargs["vocab_file"]
    merges_file = kwargs["merges_file"]
    max_length = kwargs["max_length"]

    node = [helper.make_node(
        'GPT2Tokenizer', ['string_input'], ['input_ids', 'attention_mask'], vocab=_get_file_content(vocab_file),
        merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
        domain='ai.onnx.contrib')]
    input1 = helper.make_tensor_value_info(
        'string_input', onnx_proto.TensorProto.STRING, [None])
    output1 = helper.make_tensor_value_info(
        'input_ids', onnx_proto.TensorProto.INT64, [None, None])
    output2 = helper.make_tensor_value_info(
        'attention_mask', onnx_proto.TensorProto.INT64, [None, None])

    graph = helper.make_graph(node, 'test0', [input1], [output1, output2])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)])
    return model


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokjson = _get_test_data_file('data', 'gpt2.vocab')
        cls.merges = _get_test_data_file('data', 'gpt2.merges.txt')
        cls.tokenizer = GPT2Tokenizer(cls.tokjson, cls.merges)
        cls.tokenizer.pad_token = cls.tokenizer.bos_token
        cls.max_length = -1

        @onnx_op(op_type="GPT2Tokenizer",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_int64],
                 attrs=["padding_length"])
        def bpe_tokenizer(s, **kwargs):
            # The user custom op implementation here.
            TestGPT2Tokenizer.pyop_invoked = True
            max_length = kwargs["padding_length"]
            input_ids = np.array(
                [TestGPT2Tokenizer.tokenizer(st_, max_length=max_length)["input_ids"] for st_ in s])
            attention_mask = np.array(
                [TestGPT2Tokenizer.tokenizer(st_, max_length=max_length)["attention_mask"] for st_ in s])
            return input_ids, attention_mask

    def _run_tokenizer(self, pyop_flag, test_sentence, max_length=-1):
        enable_custom_op(pyop_flag)

        if max_length == -1:
            self.max_length = None

        model = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges, max_length=max_length)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model.SerializeToString(), so)
        input_text = np.array(test_sentence)
        input_ids, attention_mask = sess.run(None, {'string_input': input_text})

        if max_length == -1:
            np.testing.assert_array_equal(input_ids, np.array(
                [TestGPT2Tokenizer.tokenizer(st_, padding=True)["input_ids"] for st_ in test_sentence]))
            np.testing.assert_array_equal(attention_mask, np.array(
                [TestGPT2Tokenizer.tokenizer(st_, padding=True)["attention_mask"] for st_ in test_sentence]))
        else:
            np.testing.assert_array_equal(input_ids, np.array(
                [TestGPT2Tokenizer.tokenizer(st_, padding="max_length", max_length=max_length)["input_ids"] for st_ in test_sentence]))
            np.testing.assert_array_equal(attention_mask, np.array(
                [TestGPT2Tokenizer.tokenizer(st_, padding="max_length", max_length=max_length)["attention_mask"] for st_ in test_sentence]))

        del sess
        del so

    def test_tokenizer(self):
        TestGPT2Tokenizer.pyop_invoked = False

        self._run_tokenizer(False, ["I can feel the magic, can you?"])
        self.assertFalse(TestGPT2Tokenizer.pyop_invoked)

        self._run_tokenizer(False, ["Hey Cortana"])
        self._run_tokenizer(False, ["你好123。david"])
        self._run_tokenizer(False, ["爱你一三一四"])
        self._run_tokenizer(False, ["women'thinsulate 3 button leather car co"])
        self._run_tokenizer(False, ["#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ"])
        self._run_tokenizer(False, ["ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ"])
        self._run_tokenizer(False, ["⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏"])
        self._run_tokenizer(False, ["I can feel the magic, can you?", "Yes I do."])

    # def test_tokenizer_pyop(self):
    #     TestGPT2Tokenizer.pyop_invoked = False
    #     self._run_tokenizer(True, ["I can feel the magic, can you?"])
    #     self.assertTrue(TestGPT2Tokenizer.pyop_invoked)
    #
    #     self._run_tokenizer(True, ["Hey Cortana"])
    #     self._run_tokenizer(True, ["你好123。david"])
    #     self._run_tokenizer(True, ["爱你一三一四"])
    #     self._run_tokenizer(True, ["women'thinsulate 3 button leather car co"])
    #     self._run_tokenizer(True, ["#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ"])
    #     self._run_tokenizer(True, ["ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ"])
    #     self._run_tokenizer(True, ["⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏"])


if __name__ == "__main__":
    unittest.main()

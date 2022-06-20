import unittest
import numpy as np
import onnxruntime as _ort

from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from transformers import GPT2Tokenizer
from onnxruntime_extensions import (
    util,
    make_onnx_model,
    enable_py_op,
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
    model = make_onnx_model(graph)
    return model


class MyGPT2Tokenizer:
    def __init__(self, token_json, merges):
        self.tokenizer = GPT2Tokenizer(token_json, merges)
        # not ensure which pad_token should be
        self.tokenizer.pad_token = '!'  # padding token = 0

    def tokenizer_sentence(self, test_sentence, padding_length):
        if padding_length == -1:
            input_ids = np.array(self.tokenizer(test_sentence, padding=True)["input_ids"])
            attention_mask = np.array(self.tokenizer(test_sentence, padding=True)["attention_mask"])
        else:
            input_ids = np.array(
                self.tokenizer(test_sentence, padding="max_length", truncation=True, max_length=padding_length)[
                    "input_ids"])
            attention_mask = np.array(
                self.tokenizer(test_sentence, padding="max_length", truncation=True, max_length=padding_length)[
                    "attention_mask"])
        return input_ids, attention_mask


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokjson = util.get_test_data_file('data', 'gpt2.vocab')
        cls.merges = util.get_test_data_file('data', 'gpt2.merges.txt')
        cls.tokenizer = MyGPT2Tokenizer(cls.tokjson, cls.merges)

        # @onnx_op(op_type="GPT2Tokenizer",
        #          inputs=[PyCustomOpDef.dt_string],
        #          outputs=[PyCustomOpDef.dt_int64, PyCustomOpDef.dt_int64],
        #          attrs=["padding_length"])
        # def bpe_tokenizer(s, **kwargs):
        #     padding_length = kwargs["padding_length"]
        #     input_ids, attention_mask = cls.tokenizer.tokenizer_sentence(s, padding_length)
        #     return input_ids, attention_mask

    def _run_tokenizer(self, test_sentence, padding_length=-1):
        model = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges, max_length=padding_length)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model.SerializeToString(), so)
        input_text = np.array(test_sentence)
        input_ids, attention_mask = sess.run(None, {'string_input': input_text})
        expect_input_ids, expect_attention_mask = self.tokenizer.tokenizer_sentence(test_sentence, padding_length)
        np.testing.assert_array_equal(expect_input_ids, input_ids)
        np.testing.assert_array_equal(expect_attention_mask, attention_mask)

        del sess
        del so

    def test_tokenizer(self):
        enable_py_op(False)

        self._run_tokenizer(["I can feel the magic, can you?"])
        self._run_tokenizer(["Hey Cortana"])
        self._run_tokenizer(["你好123。david"])
        self._run_tokenizer(["爱你一三一四"])
        self._run_tokenizer(["women'thinsulate 3 button leather car co"])
        self._run_tokenizer(["#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ"])
        self._run_tokenizer(["ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ"])
        self._run_tokenizer(["⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏"])
        self._run_tokenizer(["I can feel the magic, can you?", "Yes I do."])
        self._run_tokenizer(["I can feel the magic, can you?", "Yes I do."], 100)

        enable_py_op(True)


# def test_tokenizer_pyop(self):
#     self._run_tokenizer(["I can feel the magic, can you?"])
#     self._run_tokenizer(["Hey Cortana"])
#     self._run_tokenizer(["你好123。david"])
#     self._run_tokenizer(["爱你一三一四"])
#     self._run_tokenizer(["women'thinsulate 3 button leather car co"])
#     self._run_tokenizer(["#$%^&()!@?><L:{}\\[];',./`ǠǡǢǣǤǥǦǧǨ"])
#     self._run_tokenizer(["ڠڡڢڣڤڥڦڧڨکڪګڬڭڮگ"])
#     self._run_tokenizer(["⛀⛁⛂⛃⛄⛅⛆⛇⛈⛉⛊⛋⛌⛍⛎⛏"])


if __name__ == "__main__":
    unittest.main()

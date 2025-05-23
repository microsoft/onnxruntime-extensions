import unittest
import numpy as np
import onnxruntime as _ort

from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from transformers import RobertaTokenizer, RobertaTokenizerFast
from onnxruntime_extensions import (
    util,
    make_onnx_model,
    get_library_path as _get_library_path,
    PyOrtFunction)
from onnxruntime_extensions.cvt import HFTokenizerConverter


def _get_file_content(path):
    with open(path, "rb") as file:
        return file.read()


def _create_test_model(**kwargs):
    vocab_file = kwargs["vocab_file"]
    merges_file = kwargs["merges_file"]
    max_length = kwargs["max_length"]

    input1 = helper.make_tensor_value_info(
        'string_input', onnx_proto.TensorProto.STRING, [None])
    output1 = helper.make_tensor_value_info(
        'input_ids', onnx_proto.TensorProto.INT64, ["batch_size", "num_input_ids"])
    output2 = helper.make_tensor_value_info(
        'attention_mask', onnx_proto.TensorProto.INT64, ["batch_size", "num_attention_masks"])
    output3 = helper.make_tensor_value_info(
        'offset_mapping', onnx_proto.TensorProto.INT64, ["batch_size", "num_offsets", 2])

    if kwargs["attention_mask"]:
        if kwargs["offset_map"]:
            node = [helper.make_node(
                'RobertaTokenizer', ['string_input'], ['input_ids', 'attention_mask', 'offset_mapping'], vocab=_get_file_content(vocab_file),
                merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
                domain='ai.onnx.contrib')]

            graph = helper.make_graph(node, 'test0', [input1], [output1, output2, output3])
            model = make_onnx_model(graph)
        else:
            node = [helper.make_node(
                'RobertaTokenizer', ['string_input'], ['input_ids', 'attention_mask'], vocab=_get_file_content(vocab_file),
                merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
                domain='ai.onnx.contrib')]

            graph = helper.make_graph(node, 'test0', [input1], [output1, output2])
            model = make_onnx_model(graph)
    else:
        node = [helper.make_node(
            'RobertaTokenizer', ['string_input'], ['input_ids'], vocab=_get_file_content(vocab_file),
            merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
            domain='ai.onnx.contrib')]

        graph = helper.make_graph(node, 'test0', [input1], [output1])
        model = make_onnx_model(graph)

    return model


class TestRobertaTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model_id = util.get_test_data_file("data/tokenizer/roberta-base")
        cls.tokenizer = RobertaTokenizerFast.from_pretrained(model_id)
        temp_dir = Path('./temp_onnxroberta')
        temp_dir.mkdir(parents=True, exist_ok=True)
        files = cls.tokenizer.save_vocabulary(str(temp_dir))
        cls.tokjson = files[0]
        cls.merges = files[1]
        cls.slow_tokenizer = RobertaTokenizer.from_pretrained(model_id)
        cls.tokenizer_cvt = HFTokenizerConverter(cls.slow_tokenizer)

    def _run_tokenizer(self, test_sentence, padding_length=-1):
        model = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges,
                                   max_length=padding_length, attention_mask=True, offset_map=True)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model.SerializeToString(), so, providers=['CPUExecutionProvider'])
        input_text = np.array(test_sentence)
        input_ids, attention_mask, offset_mapping = sess.run(None, {'string_input': input_text})
        roberta_out = self.tokenizer(test_sentence, return_offsets_mapping=True)
        expect_input_ids = roberta_out['input_ids']
        expect_attention_mask = roberta_out['attention_mask']
        expect_offset_mapping = roberta_out['offset_mapping']
        np.testing.assert_array_equal(expect_input_ids, input_ids)
        np.testing.assert_array_equal(expect_attention_mask, attention_mask)
        np.testing.assert_array_equal(expect_offset_mapping, offset_mapping)

    def test_tokenizer(self):
        self._run_tokenizer([" ", "\n"])
        self._run_tokenizer(["I can feel the magic, can you?"])
        self._run_tokenizer(["Hey Cortana"])
        self._run_tokenizer(["lower newer"])
        self._run_tokenizer(["a diagram", "a dog", "a cat"])
        self._run_tokenizer(["a photo of a cat", "a photo of a dog"])
        self._run_tokenizer(["one + two = three"])
        self._run_tokenizer(["9 8 7 6 5 4 3 2 1 0"])
        self._run_tokenizer(["9 8 7 - 6 5 4 - 3 2 1 0"])
        self._run_tokenizer(["One Microsoft Way, Redmond, WA"])

    def test_converter(self):
        fn_tokenizer = PyOrtFunction.from_customop("RobertaTokenizer", cvt=(self.tokenizer_cvt).roberta_tokenizer)
        test_str = "I can feel the magic, can you?"
        fn_out = fn_tokenizer([test_str])
        roberta_out = self.tokenizer(test_str, return_offsets_mapping=True)
        expect_input_ids = roberta_out['input_ids']
        expect_attention_mask = roberta_out['attention_mask']
        expect_offset_mapping = roberta_out['offset_mapping']
        np.testing.assert_array_equal(fn_out[0].reshape((fn_out[0].size,)), expect_input_ids)
        np.testing.assert_array_equal(fn_out[1].reshape((fn_out[1].size,)), expect_attention_mask)
        np.testing.assert_array_equal(fn_out[2].reshape((fn_out[2].shape[1], fn_out[2].shape[2])), expect_offset_mapping)

    def test_optional_outputs(self):
        # Test for models without offset mapping and without both attention mask and offset mapping (input id output is always required)
        model1 = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges, max_length=-1, attention_mask=True, offset_map=False)
        model2 = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges, max_length=-1, attention_mask=False, offset_map=False)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess1 = _ort.InferenceSession(model1.SerializeToString(), so, providers=['CPUExecutionProvider'])
        sess2 = _ort.InferenceSession(model2.SerializeToString(), so, providers=['CPUExecutionProvider'])
        input_text = np.array(["Hello World"])
        outputs1 = sess1.run(None, {'string_input': input_text})
        outputs2 = sess2.run(None, {'string_input': input_text})

        # Test output size
        np.testing.assert_array_equal(len(outputs1), 2)
        np.testing.assert_array_equal(len(outputs2), 1)

        # Test output values
        roberta_out = self.tokenizer(["Hello World"], return_offsets_mapping=True)
        expect_input_ids = roberta_out['input_ids']
        expect_attention_mask = roberta_out['attention_mask']
        np.testing.assert_array_equal(expect_input_ids, outputs1[0])
        np.testing.assert_array_equal(expect_attention_mask, outputs1[1])


if __name__ == "__main__":
    unittest.main()

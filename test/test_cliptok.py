import unittest
import numpy as np
import numpy.lib.recfunctions as nlr
import onnxruntime as _ort

from pathlib import Path
from onnx import helper, onnx_pb as onnx_proto
from transformers import CLIPTokenizer, CLIPTokenizerFast
from onnxruntime_extensions import (
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

    node = [helper.make_node(
        'CLIPTokenizer', ['string_input'], ['input_ids', 'attention_mask', 'offset_mapping'], vocab=_get_file_content(vocab_file),
        merges=_get_file_content(merges_file), name='bpetok', padding_length=max_length,
        domain='ai.onnx.contrib')]
    input1 = helper.make_tensor_value_info(
        'string_input', onnx_proto.TensorProto.STRING, [None])
    output1 = helper.make_tensor_value_info(
        'input_ids', onnx_proto.TensorProto.INT64, ["batch_size", "num_input_ids"])
    output2 = helper.make_tensor_value_info(
        'attention_mask', onnx_proto.TensorProto.INT64, ["batch_size", "num_attention_masks"])
    output3 = helper.make_tensor_value_info(
        'offset_mapping', onnx_proto.TensorProto.INT64, ["batch_size", "num_offsets", 2])

    graph = helper.make_graph(node, 'test0', [input1], [output1, output2, output3])
    model = make_onnx_model(graph)
    return model


class TestCLIPTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
        temp_dir = Path('./temp_onnxclip')
        temp_dir.mkdir(parents=True, exist_ok=True)
        files = cls.tokenizer.save_vocabulary(str(temp_dir))
        cls.tokjson = files[0]
        cls.merges = files[1]
        cls.slow_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        cls.tokenizer_cvt = HFTokenizerConverter(cls.slow_tokenizer)

    def _run_tokenizer(self, test_sentence, padding_length=-1):
        model = _create_test_model(vocab_file=self.tokjson, merges_file=self.merges, max_length=padding_length)
        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(model.SerializeToString(), so)
        input_text = np.array(test_sentence)
        input_ids, attention_mask, offset_mapping = sess.run(None, {'string_input': input_text})
        print("\nTest Sentence: " + str(test_sentence))
        print("\nInput IDs: " + str(input_ids))
        print("Attention Mask: " + str(attention_mask))
        # Reformat offset mapping from 3d array to 2d array of tuples before printing for readability
        reformatted_offset_mapping = nlr.unstructured_to_structured(np.array(offset_mapping)).astype('O')
        print("Offset Mapping: " + str(reformatted_offset_mapping))
        clip_out = self.tokenizer(test_sentence, return_offsets_mapping=True)
        expect_input_ids = clip_out['input_ids']
        expect_attention_mask = clip_out['attention_mask']
        expect_offset_mapping = clip_out['offset_mapping']
        print("\nExpected Input IDs: " + str(expect_input_ids))
        print("Expected Attention Mask: " + str(expect_attention_mask))
        print("Expected Offset Mapping: " + str(expect_offset_mapping) + "\n")
        np.testing.assert_array_equal(expect_input_ids, input_ids)
        np.testing.assert_array_equal(expect_attention_mask, attention_mask)

        del sess
        del so

    def test_tokenizer(self):
        self._run_tokenizer(["I can feel the magic, can you?"])
        self._run_tokenizer(["Hey Cortana"])
        self._run_tokenizer(["lower newer"])
        self._run_tokenizer(["a diagram", "a dog", "a cat"])
        self._run_tokenizer(["A diagraM", "A DOG", "a CAT"])
        self._run_tokenizer(["a photo of a cat", "a photo of a dog"])
        self._run_tokenizer(["one + two = three"])
        self._run_tokenizer(["9 8 7 6 5 4 3 2 1 0"])
        self._run_tokenizer(["9 8 7 - 6 5 4 - 3 2 1 0"])
        self._run_tokenizer(["One Microsoft Way, Redmond, WA"])

if __name__ == "__main__":
    unittest.main()

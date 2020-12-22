import os
from pathlib import Path
import unittest
import numpy as np
from onnx import helper, onnx_pb as onnx_proto
from transformers import GPT2Tokenizer
import onnxruntime as _ort
from onnxruntime_customops import (
    onnx_op,
    PyCustomOpDef,
    expand_onnx_inputs,
    get_library_path as _get_library_path)


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
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid('', 12)])
    return model


def _bind_tokenizer(model, **kwargs):
    return expand_onnx_inputs(
        model, 'input_1',
        [helper.make_node(
            'GPT2Tokenizer', ['string_input'], ['input_1'], name='bpetok', domain='ai.onnx.contrib')],
        [helper.make_tensor_value_info('string_input', onnx_proto.TensorProto.STRING, [None])],
    )


class TestGPT2Tokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tokjson = _get_test_data_file('data', 'gpt2.vocab')
        os.environ["GPT2TOKFILE"] = tokjson
        cls.tokenizer = GPT2Tokenizer(tokjson, tokjson.replace('.vocab', '.merges.txt'))

        @onnx_op(op_type="BBPETokenizer",
                 inputs=[PyCustomOpDef.dt_string],
                 outputs=[PyCustomOpDef.dt_int64])
        def bpe_toenizer(s):
            # The user custom op implementation here.
            return np.array(
                [TestGPT2Tokenizer.tokenizer.encode(st_) for st_ in s])

    def test_tokenizer(self):
        test_sentence = "I can feel the magic, can you?"
        tokenizer = TestGPT2Tokenizer.tokenizer
        indexed_tokens = tokenizer.encode(test_sentence)

        model = _create_test_model()
        binded_model = _bind_tokenizer(model)

        so = _ort.SessionOptions()
        so.register_custom_ops_library(_get_library_path())
        sess = _ort.InferenceSession(binded_model.SerializeToString(), so)
        input_text = np.array([test_sentence])
        txout = sess.run(None, {'string_input': input_text})

        np.testing.assert_array_equal(txout[0], np.array([indexed_tokens]))


if __name__ == "__main__":
    unittest.main()

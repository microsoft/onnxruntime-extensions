import onnx
import torch
import onnxruntime_extensions

from pathlib import Path
from onnxruntime_extensions import pnp, OrtPyFunction
from transformers.convert_graph_to_onnx import convert


model_path = Path("onnx/bert-base-cased.onnx")
convert(framework="pt", model="bert-base-cased", output=model_path, opset=12)

# post-processing example function
def post_processing_forward(*pred):
    return torch.softmax(pred[0], axis=2)

# mapping the BertTokenizer outputs into the onnx model inputs
def mapping_token_output(_1, _2, _3):
    return _1.unsqueeze(0), _3.unsqueeze(0), _2.unsqueeze(0)

test_sentence = ["this is a test sentence."]
ort_tok = pnp.PreHuggingFaceBert(vocab_file=
                                 onnxruntime_extensions.get_test_data_file(
                                 '../test', 'data', 'bert_basic_cased_vocab.txt'))
onnx_model = onnx.load_model(str(model_path))

# create the final onnx model which includes pre- and post- processing.
augmented_model = pnp.export(pnp.SequentialProcessingModule(ort_tok,
                             mapping_token_output, onnx_model, post_processing_forward),
                             ["this is a test sentence."],
                             opset_version=12,
                             output_path='bert_tok_all.onnx')


# test the augmented onnx model with raw string input.
model_func = OrtPyFunction.from_model('bert_tok_all.onnx')
result = model_func(test_sentence)
print(result)

import onnx
import torch
import onnxruntime_extensions

from pathlib import Path
from onnxruntime_extensions import pnp, OrtPyFunction
from transformers import AutoTokenizer
from transformers.onnx import export, FeaturesManager

# get an onnx model by converting HuggingFace pretrained model
model_name = "bert-base-cased"
model_path = Path("onnx-model/bert-base-cased.onnx")
if not model_path.exists():
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    model = FeaturesManager.get_model_from_feature("default", model_name)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="default")
    onnx_config = model_onnx_config(model.config)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    export(tokenizer,
           model=model,
           config=onnx_config,
           opset=12,
           output=model_path)


# a silly post-processing example function, demo-purpose only
def post_processing_forward(*pred):
    return torch.softmax(pred[1], dim=1)


# mapping the BertTokenizer outputs into the onnx model inputs
def mapping_token_output(_1, _2, _3):
    return _1.unsqueeze(0), _3.unsqueeze(0), _2.unsqueeze(0)


test_sentence = ["this is a test sentence."]
ort_tok = pnp.PreHuggingFaceBert(
    vocab_file=onnxruntime_extensions.get_test_data_file(
        '../test', 'data', 'bert_basic_cased_vocab.txt'))
onnx_model = onnx.load_model(str(model_path))

# create the final onnx model which includes pre- and post- processing.
augmented_model = pnp.export(pnp.SequentialProcessingModule(
                             ort_tok, mapping_token_output,
                             onnx_model, post_processing_forward),
                             test_sentence,
                             opset_version=12,
                             output_path='bert_tok_all.onnx')

# test the augmented onnx model with raw string input.
model_func = OrtPyFunction.from_model('bert_tok_all.onnx')
result = model_func(test_sentence)
print(result)

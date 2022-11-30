import onnx
import torch

from pathlib import Path
from onnxruntime_extensions import pnp, OrtPyFunction
from transformers import AutoTokenizer
from transformers.onnx import export, FeaturesManager

# get an onnx model by converting HuggingFace pretrained model
model_name = "bert-base-cased"
model_path = Path("onnx-model/bert-base-cased.onnx")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if not model_path.exists():
    if not model_path.parent.exists():
        model_path.parent.mkdir(parents=True, exist_ok=True)
    model = FeaturesManager.get_model_from_feature("default", model_name)
    model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature="default")
    onnx_config = model_onnx_config(model.config)
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
ort_tok = pnp.PreHuggingFaceBert(hf_tok=tokenizer)
onnx_model = onnx.load_model(str(model_path))


augmented_model_name = 'temp_bert_tok_all.onnx'
# create the final onnx model which includes pre- and post- processing.
augmented_model = pnp.export(pnp.SequentialProcessingModule(
                             ort_tok, mapping_token_output,
                             onnx_model, post_processing_forward),
                             test_sentence,
                             opset_version=12,
                             output_path=augmented_model_name)

# test the augmented onnx model with raw string input.
model_func = OrtPyFunction.from_model(augmented_model_name)
result = model_func(test_sentence)
print(result)

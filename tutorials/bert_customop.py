import base64
import onnx

from onnxruntime_extensions import get_library_path, make_onnx_model, pnp
from pathlib import Path
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

# Load the onnx model as encoded string
content = open(str(model_path), "rb").read()
onnx_model_b64 = base64.b64encode(base64.decodebytes(content))

# Load the corresponding vocabulary
vocab_filepath = "data/bert-base-cased.txt"
vocabulary = open(vocab_filepath, 'r', encoding='utf-8').read()

# Create a new node for the custom op
customop_node = onnx.helper.make_node(
    'BertTokenizer',
    inputs=['inputs'],
    outputs=['input_ids', 'token_type_ids', 'attention_mask'],
    model=onnx_model_b64,
    name='BertTokenizerOp',
    domain='ai.onnx.contrib',
    vocab_file=vocabulary     # Vocabulary goes as an argument to this custom node
)

mkv = onnx.helper.make_tensor_value_info
inputs = [mkv('inputs', onnx.onnx_pb.TensorProto.STRING, [None])]
outputs = [
    mkv('input_ids', onnx.onnx_pb.TensorProto.INT64, [None]),
    mkv('token_type_ids', onnx.onnx_pb.TensorProto.INT64, [None]),
    mkv('attention_mask', onnx.onnx_pb.TensorProto.INT64, [None])
]
graph = onnx.helper.make_graph([customop_node], 'bert_customop', inputs, outputs)
onnx_model_with_customop = make_onnx_model(graph, opset_version=12)

# You could export this model out to disk using pnp.export
print(str(onnx_model_with_customop))
if 'op_type: "BertTokenizer"' not in str(onnx_model_with_customop):
  raise "Failed to add BertTokenizer customop to graph"

# Let's test it in inferencing mode
import numpy
import onnxruntime

inputs = dict(inputs=numpy.array(["this is a test sentence."], dtype=numpy.object))
options = onnxruntime.SessionOptions()
options.register_custom_ops_library(get_library_path())
session = onnxruntime.InferenceSession(onnx_model_with_customop.SerializeToString(), options)
output = session.run(None, inputs)

print(output)

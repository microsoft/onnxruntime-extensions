import onnx
import torch
import onnxruntime
import onnxruntime_extensions

from pathlib import Path
from onnxruntime_extensions import pnp
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_path = "./" + model_name + ".onnx"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# set the model to inference mode
model.eval()

# Generate dummy inputs to the model. Adjust if neccessary
inputs = {
        'input_ids':   torch.randint(32, [1, 32], dtype=torch.long), # list of numerical ids for the tokenized text
        'attention_mask': torch.ones([1, 32], dtype=torch.long)      # dummy list of ones
    }

symbolic_names = {0: 'batch_size', 1: 'max_seq_llsen'}
torch.onnx.export(model,                                         # model being run
                  (inputs['input_ids'],
                   inputs['attention_mask']), 
                  model_path,                                    # where to save the model (can be a file or file-like object)
                  opset_version=11,                              # the ONNX version to export the model to
                  do_constant_folding=True,                      # whether to execute constant folding for optimization
                  input_names=['input_ids',
                               'input_mask'],                    # the model's input names
                  output_names=['output_logits'],                # the model's output names
                  dynamic_axes={'input_ids': symbolic_names,
                                'input_mask' : symbolic_names,
                                'output_logits' : symbolic_names}) # variable length axes

# The fine-tuned HuggingFace model is exported to ONNX in the code snippet above
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model_path = Path(model_name + ".onnx")

# mapping the BertTokenizer outputs into the onnx model inputs
def map_token_output(input_ids, attention_mask, token_type_ids):
    return input_ids.unsqueeze(0), token_type_ids.unsqueeze(0), attention_mask.unsqueeze(0)

# Post process the start and end logits
def post_process(*pred):
    output = torch.argmax(pred[0])
    return output

tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_tokenizer = pnp.PreHuggingFaceBert(hf_tok=tokenizer)
bert_model = onnx.load_model(str(model_path))

augmented_model = pnp.SequentialProcessingModule(bert_tokenizer, map_token_output,
                                                 bert_model, post_process)

test_input = ["This is s test sentence"]

# create the final onnx model which includes pre- and post- processing.
augmented_model = pnp.export(augmented_model,
                             test_input,
                             opset_version=12,
                             input_names=['input'],
                             output_names=['output'],
                             output_path=model_name + '-aug.onnx',
                             dynamic_axes={'input': [0], 'output': [0]})

test_input = ["I don't really like tomatoes. They are too bitter"]

# Load the model
session_options = onnxruntime.SessionOptions()
session_options.register_custom_ops_library(onnxruntime_extensions.get_library_path())
session = onnxruntime.InferenceSession('distilbert-base-uncased-finetuned-sst-2-english-aug.onnx', session_options)

# Run the model
results = session.run(["output"], {"input": test_input})

print("\nResult is: " + ("positive" if results[0] == 1 else "negative"))

import io
import numpy as np
import onnx
import os
import torch
from onnxruntime_extensions import pnp, OrtPyFunction
from transformers import BertForQuestionAnswering, BertTokenizer

# torch.onnx.export doesn't support quantized models
# ref: https://github.com/pytorch/pytorch/issues/64477
# ref: https://github.com/pytorch/pytorch/issues/28705
# ref: https://discuss.pytorch.org/t/simple-quantized-model-doesnt-export-to-onnx/90019
# ref: https://github.com/onnx/onnx-coreml/issues/478

_this_dirpath = os.path.dirname(os.path.abspath(__file__))

question1 = "Who is John's sister?"
question2 = "Where does sophia study?"
question3 = "Who is John's mom?"
question4 = "Where does John's father's wife teach?"
context = ' '.join([
  "John is a 10 year old boy.",
  "He is the son of Robert Smith.",
  "Elizabeth Davis is Robert's wife.",
  "She teaches at UC Berkeley.",
  "Sophia Smith is Elizabeth's daughter.",
  "She studies at UC Davis.",
])

max_seq_length = 512
model_name = 'bert-large-uncased-whole-word-masking-finetuned-squad'

onnx_model_path = os.path.join(_this_dirpath, 'data', model_name + '.onnx')
onnx_tokenizer_path = os.path.join(_this_dirpath, 'data', model_name + '-tokenizer.onnx')

# Create a HuggingFace Bert Tokenizer
hf_tokenizer = BertTokenizer.from_pretrained(model_name)
# Wrap it as an ONNX operator
ort_tokenizer = pnp.HfBertTokenizer(hf_tok=hf_tokenizer)

# Code to export a hugging face bert tokenizer as an onnx model,
# currently used by tests
#
# pnp.export(
#   pnp.SequentialProcessingModule(ort_tokenizer),
#   [question1, context],
#   input_names=['text'],
#   output_names=['input_ids', 'attention_mask', 'token_type_ids'],
#   opset_version=11,
#   output_path=onnx_tokenizer_path)

# Load a pretrained HuggingFace QuestionAnswering model
model = BertForQuestionAnswering.from_pretrained(model_name)
model.eval()  # Evaluate it to switch the model into inferencing mode

symbolic_names = {0: 'batch_size', 1: 'max_seq_len'}
with io.BytesIO() as strm:
  # Export the HuggingFace model to ONNX
  torch.onnx.export(
    model,
    args=(
      torch.ones(1, max_seq_length, dtype=torch.int64),
      torch.ones(1, max_seq_length, dtype=torch.int64),
      torch.ones(1, max_seq_length, dtype=torch.int64)),
    f=strm,
    input_names=[
      'input_ids',
      'input_mask',
      'segment_ids'
    ],
    output_names=[
      'start_logits',
      'end_logits'
    ],
    dynamic_axes={
      'input_ids': symbolic_names,  # variable lenght axes
      'input_mask': symbolic_names,
      'segment_ids': symbolic_names,
      'start_logits' : symbolic_names,
      'end_logits': symbolic_names
    },
    do_constant_folding=True,
    opset_version=11)

  onnx_model = onnx.load_model_from_string(strm.getvalue())

# Export the augmented model - tokenizer, rank adjustment, q/a model
augmented_onnx_model = pnp.export(
  pnp.SequentialProcessingModule(ort_tokenizer, onnx_model),
  [question1, context],
  input_names=['text'],
  output_names=['start_logits', 'end_logits'],
  opset_version=11,
  output_path=onnx_model_path)

# Test the augmented onnx model with raw string inputs.
model_func = OrtPyFunction.from_model(onnx_model_path)

for question in [question1, question2, question3, question4]:
  result = model_func([question, context])

  # Ideally, all the logic below would be implemented as part of the augmented
  # model itself using a BertTokenizerDecoder. Unfortunately, that doesn't exist
  # at the time of this writing.

  # Get the start/end scores and find the max in each. The index of the max
  # is the start/end of the answer.
  start_scores = result[0].flatten()
  end_scores = result[1].flatten()

  answer_start_index = np.argmax(start_scores)
  answer_end_index = np.argmax(end_scores)

  start_score = np.round(start_scores[answer_start_index], 2)
  end_score = np.round(end_scores[answer_end_index], 2)

  inputs = hf_tokenizer(question, context)
  input_ids = inputs['input_ids']
  tokens = hf_tokenizer.convert_ids_to_tokens(input_ids)

  # Failed?
  if (answer_start_index == 0) or (start_score < 0) or (answer_end_index <  answer_start_index):
    answer = "Sorry, I don't know!"
  else:
    answer = tokens[answer_start_index]
    for i in range(answer_start_index + 1, answer_end_index + 1):
      if tokens[i][0:2] == '##':      # ## represent words split as two tokens
        answer += tokens[i][2:]
      else:
        answer += ' ' + tokens[i]

  print('question: ', question)
  print('  answer: ', answer)
  print('')

"""
Run this script to recreate the original onnx model.
Example usage:
python openai_audio.py out_model_path.onnx
"""

from onnx import helper, numpy_helper, TensorProto

import onnx
import numpy as np
import sys

def clear_field(proto, field):
    proto.ClearField(field)
    return proto

def order_repeated_field(repeated_proto, key_name, order):
    order = list(order)
    repeated_proto.sort(key=lambda x: order.index(getattr(x, key_name)))

def make_node(op_type, inputs, outputs, name=None, doc_string=None, domain=None, **kwargs):
    node = helper.make_node(op_type, inputs, outputs, name, doc_string, domain, **kwargs)
    if doc_string == '':
        node.doc_string = ''
    order_repeated_field(node.attribute, 'name', kwargs.keys())
    return node

def make_graph(*args, doc_string=None, **kwargs):
    graph = helper.make_graph(*args, doc_string=doc_string, **kwargs)
    if doc_string == '':
        graph.doc_string = ''
    return graph

model = helper.make_model(
    opset_imports=[clear_field(helper.make_operatorsetid('', 18), 'domain')],
    graph=make_graph(
        name='OpenAIWhisperTranscribe',
        initializer=[
            # default prompt is empty
            helper.make_tensor('transcribe0/prompt', TensorProto.STRING, [1], [b""])
        ],
        inputs=[
            helper.make_tensor_value_info('auth_token', TensorProto.STRING, shape=[1]),
            helper.make_tensor_value_info('transcribe0/file', TensorProto.UINT8, shape=["bytes"]),
            helper.make_tensor_value_info('transcribe0/prompt', TensorProto.STRING, shape=["bytes"]),
        ],
        outputs=[helper.make_tensor_value_info('transcriptions', TensorProto.STRING, shape=["num_sentences"])],
        nodes=[
            make_node(
                'OpenAIAudioToText',
                # additional optional request inputs that could be added:
                #   response_format, temperature, language
                # Using a prefix for input names allows the model to have multiple nodes calling cloud endpoints.
                # auth_token does not need a prefix unless different auth tokens are used for different nodes.
                inputs=['auth_token', 'transcribe0/file', 'transcribe0/prompt'],
                outputs=['transcriptions'],
                name='OpenAIAudioToText0',
                domain='ai.onnx.contrib',
                audio_format=b'wav',  # suffix to use for generated audio filename. 
                model_uri=b'https://api.openai.com/v1/audio/transcriptions',
                model_name=b'whisper-1',
                verbose=0,
            ),
        ],
    ),
)

if __name__ == '__main__' and len(sys.argv) == 2:
    _, out_path = sys.argv
    onnx.save(model, out_path)

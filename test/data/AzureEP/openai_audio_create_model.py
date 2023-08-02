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

# This creates a model that allows the prompt and filename to be optionally provided as inputs.
# The filename can be specified to indicate a different audio type to the default value in the audio_format attribute.
model = helper.make_model(
    opset_imports=[clear_field(helper.make_operatorsetid('', 18), 'domain')],
    graph=make_graph(
        name='OpenAIWhisperTranscribe',
        initializer=[
            # add default values in the initializers to make the model inputs optional
            helper.make_tensor('transcribe0/filename', TensorProto.STRING, [1], [b""]),
            helper.make_tensor('transcribe0/prompt', TensorProto.STRING, [1], [b""])
        ],
        inputs=[
            helper.make_tensor_value_info('auth_token', TensorProto.STRING, shape=[1]),
            helper.make_tensor_value_info('transcribe0/file', TensorProto.UINT8, shape=["bytes"]),
            helper.make_tensor_value_info('transcribe0/filename', TensorProto.STRING, shape=["bytes"]),  # optional
            helper.make_tensor_value_info('transcribe0/prompt', TensorProto.STRING, shape=["bytes"]),  # optional
        ],
        outputs=[helper.make_tensor_value_info('transcription', TensorProto.STRING, shape=[1])],
        nodes=[
            make_node(
                'OpenAIAudioToText',
                # additional optional request inputs that could be added:
                #   response_format, temperature, language
                # Using a prefix for input names allows the model to have multiple nodes calling cloud endpoints.
                # auth_token does not need a prefix unless different auth tokens are used for different nodes.
                inputs=['auth_token', 'transcribe0/file', 'transcribe0/filename', 'transcribe0/prompt'],
                outputs=['transcription'],
                name='OpenAIAudioToText0',
                domain='ai.onnx.contrib',
                audio_format=b'wav',  # default audio type if filename is not specified.
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

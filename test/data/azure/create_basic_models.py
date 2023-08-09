#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx
from onnx import helper, TensorProto


def create_audio_model():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [1])
    model = helper.make_tensor_value_info('model_name', TensorProto.STRING, [1])
    response_format = helper.make_tensor_value_info('response_format', TensorProto.STRING, [-1])
    file = helper.make_tensor_value_info('file', TensorProto.UINT8, [-1])

    transcriptions = helper.make_tensor_value_info('transcriptions', TensorProto.STRING, [-1])

    invoker = helper.make_node('OpenAIAudioToText',
                               ['auth_token', 'model_name', 'response_format', 'file'],
                               ['transcriptions'],
                               domain='com.microsoft.extensions',
                               name='audio_invoker',
                               model_uri='https://api.openai.com/v1/audio/transcriptions',
                               audio_format='wav',
                               verbose=False)

    graph = helper.make_graph([invoker], 'graph', [auth_token, model, response_format, file], [transcriptions])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])

    onnx.save(model, 'openai_audio.onnx')


def create_chat_model():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    chat = helper.make_tensor_value_info('chat', TensorProto.STRING, [-1])
    response = helper.make_tensor_value_info('response', TensorProto.STRING, [-1])

    invoker = helper.make_node('AzureTextToText', ['auth_token', 'chat'], ['response'],
                               domain='com.microsoft.extensions',
                               name='chat_invoker',
                               model_uri='https://api.openai.com/v1/chat/completions',
                               verbose=False)

    graph = helper.make_graph([invoker], 'graph', [auth_token, chat], [response])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])

    onnx.save(model, 'openai_chat.onnx')


create_audio_model()
create_chat_model()

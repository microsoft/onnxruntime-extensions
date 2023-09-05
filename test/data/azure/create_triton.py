from onnx import helper, TensorProto, save
# from onnxruntime import *

def createAddf():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.FLOAT, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-5095584.westus2.inference.ml.azure.com',
                               model_name='addf', model_version='1', verbose='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addf.onnx')

def createAddf8():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.DOUBLE, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.DOUBLE, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.DOUBLE, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-3586177.westus2.inference.ml.azure.com',
                               model_name='addf8', model_version='1', verbose='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addf8.onnx')

def createAddi4():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.INT32, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.INT32, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.INT32, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-9231153.westus2.inference.ml.azure.com',
                               model_name='addi4', model_version='1', verbose='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_addi4.onnx')

def createAnd():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    X = helper.make_tensor_value_info('X', TensorProto.BOOL, [-1])
    Y = helper.make_tensor_value_info('Y', TensorProto.BOOL, [-1])
    Z = helper.make_tensor_value_info('Z', TensorProto.BOOL, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'X', 'Y'], ['Z'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-5911162.westus2.inference.ml.azure.com',
                               model_name='and', model_version='1', verbose='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, X, Y], [Z])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_and.onnx')

def createStr():
    auth_token = helper.make_tensor_value_info('auth_token', TensorProto.STRING, [-1])
    str_in = helper.make_tensor_value_info('str_in', TensorProto.STRING, [-1])
    str_out1 = helper.make_tensor_value_info('str_out1', TensorProto.STRING, [-1])
    str_out2 = helper.make_tensor_value_info('str_out2', TensorProto.STRING, [-1])
    invoker = helper.make_node('AzureTritonInvoker', ['auth_token', 'str_in'], ['str_out1','str_out2'],
                               domain='com.microsoft.extensions', name='triton_invoker',
                               model_uri='https://endpoint-7270363.westus2.inference.ml.azure.com',
                               model_name='str', model_version='1', verbose='1')
    graph = helper.make_graph([invoker], 'graph', [auth_token, str_in], [str_out1, str_out2])
    model = helper.make_model(graph,
                              opset_imports=[helper.make_operatorsetid('com.microsoft.extensions', 1)])
    save(model, 'triton_str.onnx')

createAddf()
createAddf8()
createAddi4()
createAnd()
createStr()

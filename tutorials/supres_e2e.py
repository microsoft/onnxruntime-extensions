import io
import numpy as np
import onnx
import onnxruntime
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

# ref: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html


_this_dirpath = os.path.dirname(os.path.abspath(__file__))
_data_dirpath = os.path.join(_this_dirpath, 'data')
_onnx_model_filepath = os.path.join(_data_dirpath, 'supres.onnx')

_domain = 'ai.onnx.contrib'
_opset_version = 11


class SuperResolutionNet(nn.Module):
  def __init__(self, upscale_factor, inplace=False):
    super(SuperResolutionNet, self).__init__()

    self.relu = nn.ReLU(inplace=inplace)
    self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
    self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
    self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
    self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
    self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    self._initialize_weights()

  def forward(self, x):
    x = self.relu(self.conv1(x))
    x = self.relu(self.conv2(x))
    x = self.relu(self.conv3(x))

    return self.pixel_shuffle(self.conv4(x))

  def _initialize_weights(self):
    nn.init.orthogonal_(self.conv1.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv2.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv3.weight, nn.init.calculate_gain('relu'))
    nn.init.orthogonal_(self.conv4.weight)


# Create the super-resolution model by using the above model definition.
torch_model = SuperResolutionNet(upscale_factor=3)

# Load pretrained model weights
model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1    # just a random number

# Initialize model with the pretrained weights
map_location = lambda storage, loc: storage
if torch.cuda.is_available():
  map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

# set the model to inference mode
torch_model.eval()

# Input to the model
input = torch.randn(batch_size, 1, 224, 224, requires_grad=True)

# Export the model
with io.BytesIO() as strm:
  torch.onnx.export(torch_model,                                    # model being run
                    input,                                          # model input (or a tuple for multiple inputs)
                    strm,                                           # where to save the model (can be a file or file-like object)
                    export_params=True,                             # store the trained parameter weights inside the model file
                    opset_version=10,                               # the ONNX version to export the model to
                    do_constant_folding=True,                       # whether to execute constant folding for optimization
                    input_names = ['input'],                        # the model's input names
                    output_names = ['output'],                      # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},     # variable length axes
                                  'output' : {0 : 'batch_size'}})

  onnx_model = onnx.load_model_from_string(strm.getvalue())
  onnx.checker.check_model(onnx_model)

  ort_session = onnxruntime.InferenceSession(strm.getvalue())

torch_out = torch_model(input)

def to_numpy(tensor):
  return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# compute ONNX Runtime output prediction
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outputs = ort_session.run(None, ort_inputs)

# compare ONNX Runtime and PyTorch results
np.testing.assert_allclose(to_numpy(torch_out), ort_outputs[0], rtol=1e-03, atol=1e-05)


# Generate an model pipeline with pre/post nodes
mkv = onnx.helper.make_tensor_value_info
onnx_opsetids = [
  onnx.helper.make_opsetid('', _opset_version),
  onnx.helper.make_opsetid(_domain, _opset_version)
]

# Create custom op node for pre-processing
preprocess_node = onnx.helper.make_node(
  'SuperResolutionPreProcess',
  inputs=['raw_input_image'],
  outputs=['input', 'cr', 'cb'],
  name='Preprocess',
  doc_string='Preprocessing node',
  domain=_domain)

process_model = onnx_model
process_model.opset_import.pop()
process_model.opset_import.extend(onnx_opsetids)
onnx.checker.check_model(process_model)
process_graph = process_model.graph

# Create custom op node for post-processing
postprocess_node = onnx.helper.make_node(
  'SuperResolutionPostProcess',
  inputs=['output', 'cr', 'cb'],
  outputs=['raw_output_image'],
  name='Postprocess',
  doc_string='Postprocessing node',
  domain=_domain)

inputs = [mkv('raw_input_image', onnx.onnx_pb.TensorProto.UINT8, [])]
outputs = [mkv('raw_output_image', onnx.onnx_pb.TensorProto.UINT8, [])]
nodes = [preprocess_node] + list(process_graph.node) + [postprocess_node]
graph = onnx.helper.make_graph(
  nodes, 'supres_graph', inputs, outputs,
  initializer=list(process_graph.initializer))
onnx_model = onnx.helper.make_model(graph)
onnx_model.opset_import.pop()
onnx_model.opset_import.extend(onnx_opsetids)
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

onnx_model_as_string = str(onnx_model)
onnx_model_as_text = onnx.helper.printable_graph(onnx_model.graph)

if 'op_type: "SuperResolutionPreProcess"' not in onnx_model_as_string:
  raise "Failed to add pre-process to onnx graph"

if 'op_type: "SuperResolutionPostProcess"' not in onnx_model_as_string:
  raise "Failed to add post-process to onnx graph"

if 'SuperResolutionPreProcess(%raw_input_image)' not in onnx_model_as_text:
  raise "Failed to add pre-process to onnx graph"

if 'SuperResolutionPostProcess(%output, %cr, %cb)' not in onnx_model_as_text:
  raise "Failed to add post-process to onnx graph"

onnx.checker.check_model(onnx_model)
onnx.save(onnx_model, _onnx_model_filepath)

# Test with a inferencing session
import numpy as np
from onnxruntime_extensions import OrtPyFunction

_input_image_filepath = os.path.join(_data_dirpath, 'cat_224x224.jpg')
_output_image_filepath = os.path.join(_data_dirpath, 'cat_672x672.jpg')

encoded_input_image = open(_input_image_filepath, 'rb').read()
raw_input_image = np.frombuffer(encoded_input_image, dtype=np.uint8)

model_func = OrtPyFunction.from_model(_onnx_model_filepath)
raw_output_image = model_func(raw_input_image)

encoded_output_image = raw_output_image.tobytes()
with open(_output_image_filepath, 'wb') as strm:
  strm.write(encoded_output_image)
  strm.flush()

'''
Steps to integrate the generated model in Android app:

Assuming the app is using extensions as a separate
binary rather than embedding extensions into the ORT build.

1. Drop the generated extensions binary (libortcustomops.so) into your app's
   resources folder (usually app/src/main/jniLibs/armeabi-v7a)

2. When creating an OrtSession, add the following statement to register extensions

   val options = OrtSession.SessionOptions()
   options.registerCustomOpLibrary("libortcustomops.so")
   val session = ortEnv?.createSession(model, options)

3. Call OrtSession.run to generate the output.

   val rawImageData =  // raw image data in bytes
   val shape = longArrayOf(rawImageData.size.toLong())
   val tensor = OnnxTensor.createTensor(env, ByteBuffer.wrap(rawImageData), shape, OnnxJavaType.UINT8)
   val output = session?.run(Collections.singletonMap("input", tensor))

   "output" is the jpeg encoded high resolution image.
'''

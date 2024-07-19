import io
import onnx
import unittest
import torch
import numpy as np
import onnxruntime as _ort
from onnxruntime_extensions import (
    onnx_op, PyCustomOpDef,
    get_library_path as _get_library_path)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms


def _create_test_model(device:str="cpu", seed=42):
    # Basic setup
    use_cuda = "cuda" in device.lower() and torch.cuda.is_available()
    torch.manual_seed(seed)

    device = torch.device(device)

    # Data loader stuff
    export_kwargs = {'batch_size': 1}
    if use_cuda:
        export_kwargs = {'num_workers': 1,
                         'pin_memory': True,
                         'shuffle': True}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    export_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    export_loader = torch.utils.data.DataLoader(export_dataset,**export_kwargs)

    # Register custom op for relu in onnx and use in the model
    # Domain must be "ai.onnx.contrib" to be compatible with onnxruntime-extensions
    from torch.onnx import register_custom_op_symbolic

    def com_amd_relu_1(g, input):
        return g.op("ai.onnx.contrib::MyReLu", input).setType(input.type())

    register_custom_op_symbolic("::relu", com_amd_relu_1, 9)

    # Model
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 10, 5)
            self.conv2 = nn.Conv2d(10, 20, 5)
            self.conv2_drop = nn.Dropout2d()
            self.dropout = nn.Dropout(0.5)
            self.fc1 = nn.Linear(320, 50)
            self.fc2 = nn.Linear(50, 10)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.conv1(x)
            x = torch.max_pool2d(x, 2)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.conv2_drop(x)
            x = torch.max_pool2d(x, 2)
            x = self.relu(x)
            x = x.view(-1, 320)
            x = self.fc1(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output

    # Exporting to ONNX with custom op
    model = Net().to(device)
    input_sample = next(iter(export_loader))
    input_sample[0] = input_sample[0].to(device)  # torch.randn(1,1,28,28, dtype=torch.float)
    input_sample[1] = input_sample[1].to(device)  # torch.randint(1,10,(1,))
    f = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(model, (input_sample[0],), f)
    model_proto = onnx.ModelProto.FromString(f.getvalue())
    return model_proto, input_sample

class TestPythonOp(unittest.TestCase):

    # Used to test custom op on PyThon.
    # The ONNX graph has the custom which is executed by the function below
    # @classmethod
    # def setUpClass(cls):
    #     @onnx_op(op_type="MyReLu",
    #              inputs=[PyCustomOpDef.dt_float],
    #              outputs=[PyCustomOpDef.dt_float])
    #     def myrelu(x):
    #         return torch.relu(torch.from_numpy(x.copy()))

    def test_python_myrelu(self):
        # EPs = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        EPs = ['CPUExecutionProvider']  # TODO: Test with CUDA
        DEVICEs = ["cpu", "cuda"]
        for dev_idx, ep in enumerate(EPs):
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model, inputs = _create_test_model(device=DEVICEs[dev_idx], seed=42)
            self.assertIn('op_type: "MyReLu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=[ep])
            out = sess.run(None, {'input.1': inputs[0].numpy(force=True)})

    def test_cc_myrelu(self):
        # EPs = ['CPUExecutionProvider', 'CUDAExecutionProvider']
        EPs = ['CPUExecutionProvider']  # TODO: Test with CUDA
        DEVICEs = ["cpu", "cuda"]
        for dev_idx, ep in enumerate(EPs):
            so = _ort.SessionOptions()
            so.register_custom_ops_library(_get_library_path())
            onnx_model, inputs = _create_test_model(device=DEVICEs[dev_idx], seed=42)
            self.assertIn('op_type: "MyReLu"', str(onnx_model))
            sess = _ort.InferenceSession(onnx_model.SerializeToString(), so, providers=[ep])
            out = sess.run(None, {'input.1': inputs[0].numpy(force=True)})


if __name__ == "__main__":
    unittest.main()

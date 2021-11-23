import torch
from onnx.onnx_pb import TensorProto


class ProcessingModule(torch.nn.Module):
    @staticmethod
    def _argsort(g, x, dim, descending):
        return g.op('ai.onnx.contrib::ArgSort', x, dim)

    @classmethod
    def register_customops(cls):
        if hasattr(cls, 'loaded'):
            return True

        torch.onnx.register_custom_op_symbolic('::argsort', cls._argsort, 1)
        # ... more

        cls.loaded = True
        return True

    @classmethod
    def export(self, opset_version, *args):
        return None


tensor_data_type = TensorProto

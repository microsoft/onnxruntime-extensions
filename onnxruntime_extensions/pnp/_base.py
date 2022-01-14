import io
import onnx
import torch
from typing import Any
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

    def export(self, opset_version, *args, **kwargs):
        mod = self
        script_model = kwargs.pop('script_mode', False)
        if script_model:
            mod = torch.jit.script(mod)

        ofname = kwargs.pop('ofname', None)

        with io.BytesIO() as f:
            torch.onnx.export(mod, args, f,
                              training=torch.onnx.TrainingMode.EVAL,
                              opset_version=opset_version,
                              **kwargs)

            mdl = onnx.load_model(io.BytesIO(f.getvalue()))
            if ofname is not None:
                ofname.replace('.onnx', '.1.onnx')
                onnx.save_model(mdl, ofname)
            return mdl


class CustomFunction(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        return grad_outputs

    @classmethod
    def forward(cls, ctx: Any, *args: Any, **kwargs: Any) -> Any:
        pass

    @classmethod
    def symbolic(cls, g, *args):
        return g.op('ai.onnx.contrib::' + cls.__name__, *args)


tensor_data_type = TensorProto

import io
import onnx
import torch
from typing import Any
from onnx.onnx_pb import TensorProto
from torch.onnx import TrainingMode, export as _export

from ._onnx_ops import OPSET_TO_IR_VERSION


def _export_f(model, *args,
              opset_version=None,
              output_path=None,
              output_seq=0,
              export_params=True,
              verbose=False,
              input_names=None,
              output_names=None,
              operator_export_type=None,
              do_constant_folding=True,
              dynamic_axes=None,
              keep_initializers_as_inputs=None,
              custom_opsets=None):

    with io.BytesIO() as f:
        _export(model, args, f,
                export_params=export_params, verbose=verbose,
                training=TrainingMode.EVAL, input_names=input_names,
                output_names=output_names,
                operator_export_type=operator_export_type, opset_version=opset_version,
                do_constant_folding=do_constant_folding,
                dynamic_axes=dynamic_axes,
                keep_initializers_as_inputs=keep_initializers_as_inputs,
                custom_opsets=custom_opsets)

        mdl = onnx.load_model(io.BytesIO(f.getvalue()))
        for ops in mdl.opset_import:
            if ops.domain in ('', 'ai.onnx'):
                mdl.ir_version = OPSET_TO_IR_VERSION[ops.version]
        if output_path is not None:
            if output_seq > 0:
                output_path.replace('.onnx', '.{}.onnx'.format(output_seq))
            onnx.save_model(mdl, output_path)
        return mdl


class _ProcessingModule:

    def __init__(self):
        super(_ProcessingModule, self).__init__()
        _ProcessingModule.register_customops()

    @staticmethod
    @torch.jit.unused
    def _argsort(g, x, dim, descending):
        return g.op('ai.onnx.contrib::ArgSort', x, dim)

    @classmethod
    @torch.jit.unused
    def register_customops(cls):
        if hasattr(cls, 'loaded'):
            return True

        torch.onnx.register_custom_op_symbolic('::argsort', cls._argsort, 1)
        # ... more

        cls.loaded = True
        return True

    @torch.jit.unused
    def export(self, *args, opset_version=None, script_mode=False, output_path=None, output_seq=0, **kwargs):
        if opset_version is None:
            raise RuntimeError('No opset_version found in the kwargs.')
        mod = self
        if script_mode and not isinstance(mod, torch.jit.ScriptModule):
            mod = torch.jit.script(mod)

        return _export_f(mod,
                         *args,
                         opset_version=opset_version,
                         output_path=output_path,
                         output_seq=output_seq, **kwargs)


class ProcessingTracedModule(torch.nn.Module, _ProcessingModule):
    def __init__(self, func_obj=None):
        super().__init__()
        self.func_obj = func_obj

    def forward(self, *args):
        assert self.func_obj is not None, "No forward method found."
        return self.func_obj(*args)


class ProcessingScriptModule(torch.nn.Module, _ProcessingModule):

    @torch.jit.unused
    def export(self, *args, **kwargs):
        return super().export(*args, script_mode=True, **kwargs)


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


def is_processing_module(m):
    return isinstance(m, _ProcessingModule)

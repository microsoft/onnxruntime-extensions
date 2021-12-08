import io
import onnx
import torch
import numpy
from torch.onnx import TrainingMode, export as _export
from .pnp import ONNXModelUtils, ProcessingModule
from ._ortapi2 import OrtPyFunction as ONNXPyFunction


def _is_numpy_object(x):
    return isinstance(x, (numpy.ndarray, numpy.generic))


def _is_numpy_string_type(arr):
    return arr.dtype.kind in {'U', 'S'}


def _export_f(model, args=None,
              export_params=True, verbose=False,
              input_names=None, output_names=None,
              operator_export_type=None, opset_version=None,
              do_constant_folding=True,
              dynamic_axes=None, keep_initializers_as_inputs=None, custom_opsets=None):
    if isinstance(model, ProcessingModule):
        # try to call ProcessingModule export
        m = model.export(opset_version, *args)
        if m is not None:
            return m

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
        return onnx.load_model(io.BytesIO(f.getvalue()))


class ONNXCompose:
    """
    Merge the pre/post processing Pytorch subclassing modules with the core model.
    :arg models the core model, can be an ONNX model or a PyTorch ONNX-exportable models
    :arg preprocessors the preprocessing module
    :arg postprocessors the postprocessing module
    """
    def __init__(self, models=None, preprocessors=None, postprocessors=None):
        self.models = models
        self.preprocessors = preprocessors
        self.postprocessors = postprocessors

        self.pre_args = None
        self.models_args = None
        self.post_args = None

    def export(self, opset_version, output_file=None,
               export_params=True,
               verbose=False,
               input_names=None,
               output_names=None,
               operator_export_type=None,
               do_constant_folding=True,
               dynamic_axes=None,
               keep_initializers_as_inputs=None,
               custom_opsets=None,
               io_mapping=None):
        """
        export all models and modules into a merged ONNX model.
        """
        post_m = None
        pre_m = _export_f(self.preprocessors, tuple(self.pre_args),
                          export_params=export_params, verbose=verbose, opset_version=opset_version)

        if isinstance(self.models, torch.nn.Module):
            core = _export_f(self.models, tuple(self.models_args),
                             export_params=export_params, verbose=verbose, input_names=input_names,
                             output_names=output_names,
                             operator_export_type=operator_export_type, opset_version=opset_version,
                             do_constant_folding=do_constant_folding,
                             dynamic_axes=dynamic_axes,
                             keep_initializers_as_inputs=keep_initializers_as_inputs,
                             custom_opsets=custom_opsets)
        else:
            core = self.models

        if self.postprocessors is not None:
            post_m = _export_f(self.postprocessors, tuple(self.post_args),
                               export_params=export_params, verbose=verbose, opset_version=opset_version)
        model_l = [core]
        if pre_m is not None:
            model_l.insert(0, pre_m)
        if post_m is not None:
            model_l.append(post_m)

        full_m = ONNXModelUtils.join_models(*model_l, io_mapping=io_mapping)
        if output_file is not None:
            onnx.save_model(full_m, output_file)
        return full_m

    def predict(self, *args, extra_args_post=None):
        """
        Predict the result through all modules/models
        :param args: the input arguments for the first preprocessing module.
        :param extra_args_post: extra args for post-processors.
        :return: the result from the last postprocessing module or
                 from the core model if there is no postprocessing module.
        """
        def _is_tensor(x):
            if isinstance(x, list):
                return all(_is_tensor(_x) for _x in x)
            return isinstance(x, torch.Tensor)

        def _is_array(x):
            if isinstance(x, list):
                return all(_is_array(_x) for _x in x)
            return _is_numpy_object(x) and (not _is_numpy_string_type(x))

        # convert the raw value, and special handling for string.
        n_args = [numpy.array(_arg) if not _is_tensor(_arg) else _arg for _arg in args]
        n_args = [torch.from_numpy(_arg) if
                  _is_array(_arg) else _arg for _arg in n_args]

        self.pre_args = n_args
        inputs = [self.preprocessors.forward(*n_args)]
        flatten_inputs = []
        for _i in inputs:
            flatten_inputs += list(_i) if isinstance(_i, tuple) else [_i]
        self.models_args = flatten_inputs
        if isinstance(self.models, torch.nn.Module):
            outputs = self.models.forward(*flatten_inputs)
        else:
            f = ONNXPyFunction.from_model(self.models)
            outputs = [torch.from_numpy(f(*[_i.numpy() for _i in flatten_inputs]))]
        self.post_args = outputs
        if extra_args_post:
            if extra_args_post[0]:
                extra_args = n_args[extra_args_post[0][0]:extra_args_post[0][1]]
            if len(extra_args_post) > 1:
                extra_args += flatten_inputs[extra_args_post[1][0]:extra_args_post[1][1]]
            self.post_args = extra_args + self.post_args

        if self.postprocessors is None:
            return outputs

        return self.postprocessors.forward(*self.post_args)

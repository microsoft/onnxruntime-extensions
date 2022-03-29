import onnx

from .._ortapi2 import get_opset_version_from_ort
from ._utils import ONNXModelUtils
from ._base import is_processing_module
from ._torchext import get_id_models, SequentialProcessingModule


def export(m, *args,
           opset_version=0,
           output_path=None,
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
    if opset_version == 0:
        opset_version = get_opset_version_from_ort()

    if not is_processing_module(m):
        m = SequentialProcessingModule(m)

    model = m.export(*args, opset_version=opset_version,
                     output_path=output_path,
                     export_params=export_params,
                     verbose=verbose,
                     input_names=input_names,
                     output_names=output_names,
                     operator_export_type=operator_export_type,
                     do_constant_folding=do_constant_folding,
                     dynamic_axes=dynamic_axes,
                     keep_initializers_as_inputs=keep_initializers_as_inputs,
                     custom_opsets=custom_opsets)
    full_m = ONNXModelUtils.unfold_model(model, get_id_models(), io_mapping)
    if output_path is not None:
        onnx.save_model(full_m, output_path)
    return full_m

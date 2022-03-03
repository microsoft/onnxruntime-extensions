from ._utils import ONNXModelUtils
from ._torchext import get_id_models


def export(m, *args,
           opset_version,
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

    return ONNXModelUtils.unfold_model(model, get_id_models(), io_mapping)

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import onnx

from dataclasses import dataclass
from typing import List, Union


def create_named_value(name: str, data_type: int, shape: List[Union[str, int]]):
    """
    Helper to create a new model input.

    Args:
        name: Name for input. Must not already be in use in the model being updated.
        data_type: onnx.TensorProto data type. e.g. onnx.TensorProto.FLOAT, onnx.TensorProto.UINT8
        shape: Input shape. Use int for dimensions with known values and strings for symbolic dimensions.
               e.g. ['batch_size', 256, 256] would be a rank 3 tensor with a symbolic first dimension named 'batch_size'


    Returns:
        An onnx.ValueInfoProto that can be used as a new model input.
    """
    tensor_type = onnx.helper.make_tensor_type_proto(
        elem_type=data_type, shape=shape)
    return onnx.helper.make_value_info(name, tensor_type)


# Create an onnx checker context that includes the ort-ext domain so that custom ops don't cause failure
def create_custom_op_checker_context(onnx_opset: int):
    """
    Create an ONNX checker context that includes the ort-extensions custom op domains so that custom ops don't
    cause failure when running onnx.checker.check_graph.

    Args:
        onnx_opset: ONNX opset to use in the checker context.

    Returns:
        ONNX checker context.
    """
    context = onnx.checker.C.CheckerContext()
    context.ir_version = onnx.checker.DEFAULT_CONTEXT.ir_version
    context.opset_imports = {"": onnx_opset, "com.microsoft.extensions": 1}

    return context


# The ONNX graph parser has it's own map of names just to be special
# https://github.com/onnx/onnx/blob/604af9cb28f63a6b9924237dcb91530649233db9/onnx/defs/parser.h#L72
TENSOR_TYPE_TO_ONNX_TYPE = {
    int(onnx.TensorProto.FLOAT): "float",
    int(onnx.TensorProto.UINT8): "uint8",
    int(onnx.TensorProto.INT8): "int8",
    int(onnx.TensorProto.UINT16): "uint16",
    int(onnx.TensorProto.INT16): "int16",
    int(onnx.TensorProto.INT32): "int32",
    int(onnx.TensorProto.INT64): "int64",
    int(onnx.TensorProto.STRING): "string",
    int(onnx.TensorProto.BOOL): "bool",
    int(onnx.TensorProto.FLOAT16): "float16",
    int(onnx.TensorProto.DOUBLE): "double",
    int(onnx.TensorProto.UINT32): "uint32",
    int(onnx.TensorProto.UINT64): "uint64",
    int(onnx.TensorProto.COMPLEX64): "complex64",
    int(onnx.TensorProto.COMPLEX128): "complex128",
    int(onnx.TensorProto.BFLOAT16): "bfloat16",
}


@dataclass
class IoMapEntry:
    """Entry to map the output index from a producer step to the input index of a consumer step."""

    # optional producer
    #   Uses Step if provided.
    #   If a str with a previous Step name is provided the PrePostProcessor will find the relevant Step
    #   If neither are provided the producer is inferred to be the immediately previous Step in the pipeline
    producer: Union["Step", str] = None
    # output index from the producer step
    producer_idx: int = 0
    # input index of the consumer step
    consumer_idx: int = 0


@dataclass
class IOEntryValuePreserver:
    """
    used to allow an output value to have multiple consumers, 
    which is only possible when IoMapEntry is used to create those additional connections.

    Generally, a connection consumes an output and an input, then the output is removed from the graph.
    This class enabled one-to-many connections by making the other consumers share the same output.
    
    How this class works:
        1. when the IoMapEntry is created, this class will be created simultaneously.
        2. It records the producer and consumer steps, and the output index of the producer step.
    when producer step is running, this IOEntryValuePreserver will be activated and start to preserve the output.
        3. when graph merge happens, this class will check if the output is still in the graph, if not, 
    it will add the output
        4. when consumer step is running, this class will be deactivated and remove output from preserved_list.
    """

    producer: Union["Step", str] = None
    consumer: Union["Step", str] = None
    # output index from the producer step
    producer_idx: int = 0
    is_active: bool = False
    output: str = None


def sanitize_output_names(graph: onnx.GraphProto):
    """
    Convert any usage of invalid characters like '/' and ';' in value names to '_'
    This is common in models exported from TensorFlow [Lite].

    ONNX parse_graph does not allow for that in a value name, and technically it's a violation of the ONNX spec as per
    https://github.com/onnx/onnx/blob/main/docs/IR.md#names-within-a-graph

    We do this for the original graph outputs only. The invalid naming has not been seen in model inputs, and we can
    leave the internals of the graph intact to minimize changes.

    Args:
        graph: Graph to check and update any invalid names
    """

    bad_output_names = [o.name for o in graph.output if "/" in o.name or ";" in o.name]
    if not bad_output_names:
        return graph

    renames = {}
    for n in bad_output_names:
        renames[n] = n.replace("/", "_").replace(";", "_")

    for o in graph.output:
        if o.name in bad_output_names:
            # Add Identity node to rename the output, and update the name in graph.output
            rename = onnx.helper.make_node("Identity", [o.name], [renames[o.name]], f"Rename {o.name}")
            graph.node.append(rename)
            o.name = renames[o.name]

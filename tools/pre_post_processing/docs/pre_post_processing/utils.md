Module pre_post_processing.utils
================================

Functions
---------

    
`create_custom_op_checker_context()`
:   Create an ONNX checker context that includes the ort-extensions custom op domains so that custom ops don't
    cause failure when running onnx.checker.check_graph.
    Returns:

    
`create_named_value(name: str, data_type: int, shape: List[Union[str, int]])`
:   Helper to create a new model input.
    
    Args:
        name: Name for input. Must not already be in use in the model being updated.
        data_type: onnx.TensorProto data type. e.g. onnx.TensorProto.FLOAT, onnx.TensorProto.UINT8
        shape: Input shape. Use int for dimensions with known values and strings for symbolic dimensions.
               e.g. ['batch_size', 256, 256] would be a rank 3 tensor with a symbolic first dimension named 'batch_size'
    
    
    Returns:
        An onnx.ValueInfoProto that can be used as a new model input.

    
`get_opset_imports()`
:   Get the opset imports for a model updated by the PrePostProcessor.

    
`sanitize_output_names(graph: onnx.onnx_ml_pb2.GraphProto)`
:   Convert any usage of invalid characters like '/' and ';' in value names to '_'
    This is common in models exported from TensorFlow [Lite].
    
    ONNX parse_graph does not allow for that in a value name, and technically it's a violation of the ONNX spec as per
    https://github.com/onnx/onnx/blob/main/docs/IR.md#names-within-a-graph
    
    We do this for the original graph outputs only. The invalid naming has not been seen in model inputs, and we can
    leave the internals of the graph intact to minimize changes.
    
    Args:
        graph: Graph to check and update any invalid names

Classes
-------

`IoMapEntry(producer: Union[ForwardRef('Step'), str] = None, producer_idx: int = 0, consumer_idx: int = 0)`
:   Entry to map the output index from a producer step to the input index of a consumer step.

    ### Class variables

    `consumer_idx: int`
    :

    `producer: Union[Step, str]`
    :

    `producer_idx: int`
    :
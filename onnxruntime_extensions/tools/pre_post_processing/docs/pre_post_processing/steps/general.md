Module pre_post_processing.steps.general
========================================

Classes
-------

`ArgMax(name: Optional[str] = None, axis: int = -1, keepdims: int = 0)`
:   Base class for a pre or post processing step.
    
    Brief:
        Same as ArgMax op.
    Args:
        name: Optional name of step. Defaults to 'ArgMax'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Identity(num_inputs: int = 1, name: Optional[str] = None)`
:   ONNX Identity for all inputs to the Step. Used to pass through values as-is to later Steps.
    
    Args:
        name: Optional name of step. Defaults to 'Identity'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`ReverseAxis(axis: int = -1, dim_value: int = -1, name: Optional[str] = None)`
:   Reverses the data in an axis by splitting and concatenating in reverse order.
      e.g. convert RGB ordered data to BGR.
    Output data type and shape is the same as the input.
    
    Args:
        axis: Axis to reverse. Default is last axis.
        dim_value: Explicit value for size of dimension being reversed.
                   This can be provided if the axis being reversed currently has a symbolic value.
                   Note that this will fail during graph execution if the actual value at runtime does not match.
                   If not provided, the size of the dimension to reverse is inferred from the input shape.
        name: Optional Step name. Defaults to 'ReverseAxis'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Softmax(name: Optional[str] = None)`
:   ONNX Softmax
    
    Args:
        name: Optional Step name. Defaults to 'Softmax'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Split(num_outputs: int, axis: Optional[int] = None, splits: Optional[List[int]] = None, name: Optional[str] = None)`
:   ONNX Split
    
    :param num_outputs: Number of outputs to split the input into. Unequal split is allowed for opset 18+.
    :param axis: Axis to split on. Default is 0.
    :param splits: Optional length of each output. Sum must equal dim value at 'axis'
    :param name: Optional Step name. Defaults to 'Split'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Squeeze(axes: Optional[List[int]] = None, name: Optional[str] = None)`
:   ONNX Squeeze
    
    Args:
        axes: Axes to remove.
              If None, remove all axes with size of 1. Requires all dimensions to have explicit values.
        name: Optional Step name. Defaults to 'Squeeze'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Transpose(perms: List[int], name: Optional[str] = None)`
:   ONNX Transpose.
    
    Args:
        perms: List of integers with permutations to apply.
        name: Optional Step name. Defaults to 'Transpose'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

    ### Descendants

    * pre_post_processing.steps.vision.ChannelsLastToChannelsFirst

`Unsqueeze(axes: List[int], name: Optional[str] = None)`
:   ONNX Unsqueeze
    
    Args:
        axes: List of integers indicating the dimensions to be inserted.
        name: Optional Step name. Defaults to 'Unsqueeze'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step
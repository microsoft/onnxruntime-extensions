Module pre_post_processing.steps.general
========================================

Classes
-------

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
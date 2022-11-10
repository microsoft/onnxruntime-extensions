Module pre_post_processing.Steps.general
========================================

Classes
-------

`ReverseAxis(axis: int = -1, dim_value: int = -1, name: str = None)`
:   Reverses the data in an axis by splitting and concatenating in reverse order.    
      e.g. convert RGB ordered data to BGR. 
    Output data type and shape is the same as the input. 
    
    Initialize ReverseAxis step.
    Args:
        axis: Axis to reverse. Default is last axis.
        dim_value: Explicit value for size of dimension being reversed.
                   This can be provided if the axis being reversed currently has a symbolic value.
                   Note that this will fail during graph execution if the actual value at runtime does not match.
                   If not provided, the size of the dimension to reverse is inferred from the input shape.
        name: Optional Step name. Defaults to 'ReverseAxis'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Softmax(name: str = None)`
:   ONNX Softmax
    
    Initialize the step.
    
    Args:
        inputs: List of default input names.
        outputs: List of default output names.
        name: Step name. Defaults to the derived class name.

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Squeeze(axes: List[int] = None, name: str = None)`
:   ONNX Squeeze
    
    Initialize the step.
    
    Args:
        inputs: List of default input names.
        outputs: List of default output names.
        name: Step name. Defaults to the derived class name.

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Transpose(perms: List[int], name: str = None)`
:   ONNX Transpose.
    
    Initialize the step.
    
    Args:
        inputs: List of default input names.
        outputs: List of default output names.
        name: Step name. Defaults to the derived class name.

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

    ### Descendants

    * pre_post_processing.Steps.vision.ChannelsLastToChannelsFirst

`Unsqueeze(axes: List[int], name: str = None)`
:   ONNX Unsqueeze
    
    Initialize the step.
    
    Args:
        inputs: List of default input names.
        outputs: List of default output names.
        name: Step name. Defaults to the derived class name.

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step
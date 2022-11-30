Module pre_post_processing.step
===============================

Classes
-------

`Debug(num_inputs: int = 1, name: Optional[str] = None)`
:   Step that can be arbitrarily inserted in the pre or post processing pipeline.
    It will make the outputs of the previous Step also become graph outputs so their value can be more easily debugged.
    
    NOTE: Depending on when the previous Step's outputs are consumed in the pipeline the graph output for it
          may or may not have '_debug' as a suffix.
          TODO: PrePostProcessor __cleanup_graph_output_names could also hide the _debug by inserting an Identity node
                to rename so it's more consistent.
    
    Initialize Debug step
    Args:
        num_inputs: Number of inputs from previous Step to make graph outputs.
        name: Optional name for Step. Defaults to 'Debug'

    ### Ancestors (in MRO)

    * pre_post_processing.step.Step

`Step(inputs: List[str], outputs: List[str], name: Optional[str] = None)`
:   Base class for a pre or post processing step.
    
    Initialize the step.
    
    Args:
        inputs: List of default input names.
        outputs: List of default output names.
        name: Step name. Defaults to the derived class name.

    ### Descendants

    * pre_post_processing.step.Debug
    * pre_post_processing.steps.general.ReverseAxis
    * pre_post_processing.steps.general.Softmax
    * pre_post_processing.steps.general.Squeeze
    * pre_post_processing.steps.general.Transpose
    * pre_post_processing.steps.general.Unsqueeze
    * pre_post_processing.steps.vision.CenterCrop
    * pre_post_processing.steps.vision.ConvertBGRToImage
    * pre_post_processing.steps.vision.ConvertImageToBGR
    * pre_post_processing.steps.vision.FloatToImageBytes
    * pre_post_processing.steps.vision.ImageBytesToFloat
    * pre_post_processing.steps.vision.Normalize
    * pre_post_processing.steps.vision.PixelsToYCbCr
    * pre_post_processing.steps.vision.Resize
    * pre_post_processing.steps.vision.YCbCrToPixels

    ### Class variables

    `prefix`
    :

    ### Methods

    `apply(self, graph: onnx.onnx_ml_pb2.GraphProto)`
    :   Append the nodes that implement this step to the provided graph.

    `connect(self, entry: pre_post_processing.utils.IoMapEntry)`
    :   Connect the value name from a previous step to an input of this step so they match.
        This makes joining the GraphProto created by each step trivial.
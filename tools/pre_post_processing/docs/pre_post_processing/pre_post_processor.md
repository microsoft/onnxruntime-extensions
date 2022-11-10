Module pre_post_processing.pre_post_processor
=============================================

Classes
-------

`PrePostProcessor(inputs: List[onnx.onnx_ml_pb2.ValueInfoProto] = None, outputs: List[onnx.onnx_ml_pb2.ValueInfoProto] = None)`
:   Class to handle running all the pre/post processing steps and updating the model.

    ### Methods

    `add_post_processing(self, items: List[Union[pre_post_processing.step.Step, Tuple[pre_post_processing.step.Step, List[pre_post_processing.utils.IoMapEntry]]]])`
    :

    `add_pre_processing(self, items: List[Union[pre_post_processing.step.Step, Tuple[pre_post_processing.step.Step, List[pre_post_processing.utils.IoMapEntry]]]])`
    :   Add the pre-processing steps.
        Options are:
          Add Step with default connection of outputs from the previous step (if available) to inputs of this step.
          Add tuple of Step or step name and io_map for connections between two steps.
            If IoMapEntry.producer is not specified it is inferred to be the previous Step.

    `run(self, model: onnx.onnx_ml_pb2.ModelProto)`
    :   Update the model with the graph from each step in the pre and post processing pipelines.
        
        Args:
            model: model to add pre/post processing to.
        
        Returns:
            model with pre/post processing in it.
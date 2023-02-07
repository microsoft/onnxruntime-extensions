Module pre_post_processing.step
===============================

Classes
-------

`Debug(num_inputs: int = 4, name: Optional[str] = None, custom_func: Optional[<built-in function callable>] = None)`
:   Step that can be arbitrarily inserted in the pre or post processing pipeline.
    It will make the outputs of the previous Step also become graph outputs so their value can be more easily debugged.
    
    We will duplicate the outputs of graph, the original outputs will be duplicated, one will be renamed with a suffix "_next",
    another will be renamed with a suffix "_debug".the "_next" outputs will feed into the next step,
    the "_debug" outputs will become graph outputs.
    
    Initialize Debug step
    Args:
        num_inputs: Number of inputs from previous Step to make graph outputs. Devs can set any number of inputs to be debugged.
            (named inputs are not supported though). This class will handle it if the number of inputs is less than the number.
        name: Optional name for Step. Defaults to 'Debug'
        custom_func: Optional custom function to visit the graph, A very simple example is to save the graph to a file.
            For example:
                ```
                def save_onnx(graph):
                    opset_imports = [
                        onnx.helper.make_operatorsetid(domain, opset)
                        for domain, opset in pipeline._custom_op_checker_context.opset_imports.items()
                    ]
                    new_model = onnx.helper.make_model(graph, opset_imports=opset_imports)
                    onnx.save_model(new_model, "debug.onnx")
                Debug(custom_func=save_onnx)
                ```

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
    * pre_post_processing.steps.nlp.BertTokenizer
    * pre_post_processing.steps.nlp.BertTokenizerQATask
    * pre_post_processing.steps.nlp.BertTokenizerQATaskDecoder
    * pre_post_processing.steps.nlp.SentencePieceTokenizer
    * pre_post_processing.steps.nlp.SequenceClassify
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

    `apply(self, graph: onnx.onnx_ml_pb2.GraphProto, checker_context: onnx.onnx_cpp2py_export.checker.CheckerContext)`
    :   Create a graph for this step that can be appended to the provided graph.
        The PrePostProcessor will handle merging the two.

    `connect(self, entry: pre_post_processing.utils.IoMapEntry)`
    :   Connect the value name from a previous step to an input of this step so they match.
        This makes joining the GraphProto created by each step trivial.
# Example usage of the PrePostProcessor

The PrePostProcessor can be used to add pre and post processing operations to an existing model.

Currently the easiest way to use it is to download this folder and import PrePostProcessor and the Steps into your python script.
We will provide a python package that includes it in the next release.


## Initial imports

Import the PrePostProcessor, the steps and a utility to simplify creating new model inputs.

```py
import onnx
from pre_post_processing import PrePostProcessor
from pre_post_processing.Steps import *
from pre_post_processing.utils import create_named_value
```

## Example of creating the pre and post processing pipelines

The following is an example pre-processing pipeline to update a model to take bytes from an jpg or png image as input.
The original model input was pre-processed float data with shape {1, channels, 244, 244}, requiring the user to
manually convert their input image to this format.

### Create new input/s for the model

First, if you're adding pre-processing you need to create new inputs to the model that the pre-processing will use.

In our example we'll create a new input called 'image' containing uint8 data of length 'num_bytes'.

```py
new_input = create_named_value('image', onnx.TensorProto.UINT8, ['num_bytes'])
```

### Create PrePostProcessor

Create our PrePostProcessor instance with the new input/s.

```py
pipeline = PrePostProcessor([new_input])
```

### Add pre-processing steps

Add the preprocessing steps to the PrePostProcessor in the desired order.
You can pick-and-choose from the predefined steps in the pre_post_processing.Steps module or create your own custom steps.
If there's some common pre or post processing functionality that is missing please reach out and we'll look at adding
the necessary Step implementations for it.

Configure the steps as needed.

```py
pipeline.add_pre_processing(
    [
        ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout. output shape is {h_in, w_in, channels}
        Resize(256),          # resize so smallest side is 256.
        CenterCrop(224, 224),
        ChannelsLastToChannelsFirst(),  # ONNX models are typically channels first. output shape is {channels, 244, 244}
        ImageBytesToFloat(),  # Convert uint8 values in range 0..255 to float values in range 0..1
        Unsqueeze(axes=[0]),  # add batch dim so shape is {1, channels, 244, 244}. we now match the original model input
    ]
)
```

Outputs from the previous step will be automatically connected to the next step (or model in the case of the last step),
in the same order.
i.e. the first output of the previous step is connected to the first input of the next step, etc. etc.
until we run out of outputs or inputs (whichever happens first).

It is also possible to manually specify connections. See [IoMapEntry](#iomapentry_usage)


### Add post-processing steps

Similarly the post-processing is assembled the same way. Let's say it's simply a case of applying Softmax to the
first model output:

``` py
pipeline.add_pre_processing(
    [
        Softmax()
    ]
)
```

Neither pre-processing or post-processing is required. Simply add what you need for your model.

### Execute pipeline

Once we have assembled our pipeline we simply run it with the original model, and save the output.

The last pre-processing step is automatically connected to the original model inputs,
and the first post-processing step is automatically connected to the original model outputs.

```py
model = onnx.load('my_model.onnx')
new_model = pipeline.run(model)
onnx.save_model(new_model, 'my_model.with_pre_post_processing.onnx')
```


## Helper to create new named model inputs.

The `create_named_value` helper from [pre_post_processing.utils](./docs/pre_post_processing/utils.md#) can be used
to create model inputs.

- The `name` value must be unique for the model.
- The `data_type` should be an onnx.TensorProto value like onnx.TensorProto.UINT8 or onnx.TensorProto.FLOAT from the
list defined [here](https://github.com/onnx/onnx/blob/759907808db622938082c6eeaa8f685dee3dc868/onnx/onnx.proto#L483).
- The `shape` specifies the input shape. Use int for dimensions with known values and strings for symbolic dimensions.
  e.g. ['batch_size', 1024] would be a rank 2 tensor with a symbolic first dimension named 'batch_size'.


## IoMapEntry usage

When the automatic connection of outputs from the previous step to inputs of the current step is insufficient,
an IoMapEntry can be used to explicitly specify connections.

As an example, let's look at a subset of the operations in the pre and post processing for a super resolution model.
In the pre-processing we convert the input from RGB to YCbCr using `PixelsToYCbCr`.
That step produces 3 separate outputs - `Y`, `Cb` and `Cr`. The model has one input and is automatically connected
to the `Y` output when PixelsToYCbCr is the last pre-processing step.
We want to consume the `Cr` and `Cr` outputs in the post-processing by joining that with new `Y'` model output.


```py
    pipeline = PrePostProcessor(inputs)
    pipeline.add_pre_processing(
        [
            ...
            # this produces Y, Cb and Cr outputs. each has shape {h_in, w_in}. only Y is input to model
            PixelsToYCbCr(layout="BGR"),
        ]
    )
```

In order to do that, the post-processing entry can be specified as a tuple of the Step and a list of IoMapEntries.
Each IoMapEntry has a simple structure of `IoMapEntry(producer, producer_idx, consumer_idx)`. The `producer` is the
name of the Step that produces the output. The `producer_idx` is the index of the output from that step. The `consumer_idx`
is the input number of the Step that we want to connect to.


```py
    pipeline.add_post_processing(
        [
            # as we're selecting outputs from multiple previous steps we need to map them to the inputs using step names
            (
                YCbCrToPixels(layout="BGR"),
                [
                    # the first model output is automatically joined to consumer_idx=0
                    IoMapEntry("PixelsToYCbCr", producer_idx=1, consumer_idx=1),  # Cb value
                    IoMapEntry("PixelsToYCbCr", producer_idx=2, consumer_idx=2)   # Cr value
                ],
            ),
            ConvertBGRToImage(image_format="png")
        ]
    )
```

By default the name for the each Step is the class name. When instantiating a step you can override the `name` property
to provide a more descriptive name or resolve ambiguity (e.g. if there are multiple steps of the same type).

In our example, if we used `PixelsToYCbCr(layout="BGR", name="ImageConverter")` in the pre-processing step,
we would use `IoMapEntry("ImageConverter", producer_idx=1, consumer_idx=1)` in the post-processing step to match that
name.

Note that the automatic connection between steps will still occur. The list of IoMapEntry values is used to override the
automatic connections, so you only need to provide an IoMapEntry for connections that need customization. In our
example the model output is automatically connected to the first input of the `YCbCrToPixels` step so it wasn't
necessary to provide an IoMapEntry for consumer_idx=0.


## Debug step usage

If you are creating your own pipeline if can sometimes be necessary to inspect the output of a pre or post processing
step if the final results are unexpected. The easiest way to do this is to insert a `Debug` step into the pipeline.

The Debug step will create graph outputs for the outputs from the previous step. That means they will be available
as outputs when running the updated model, and can be inspected.

The Debug step will also pass through its inputs to the next step, so no other changes to the pipeline are required.

Considering our pre-processing example, if we wanted to inspect the result of the conversion from an input image
we can insert a Debug step like below. The existing steps remain unchanged.

```py
pipeline.add_pre_processing(
    [
        ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout. output shape is {h_in, w_in, channels}
        Debug(),
        Resize(256),          # resize so smallest side is 256.
```

The model will now have an additional output called 'bgr_data' (the default output name of the ConvertImageToBGR step).

Note that if the previous step produces multiple outputs the Debug step must be configured with this information.

e.g.

```py
PixelsToYCbCr(layout="BGR"),
Debug(num_inputs=3),
...
```

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import enum
import os

from pathlib import Path

from pre_post_processing import PrePostProcessor
from pre_post_processing.steps import *
from pre_post_processing.utils import create_named_value, IoMapEntry


class ModelSource(enum.Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    OTHER = 2


def imagenet_preprocessing(model_source: ModelSource = ModelSource.PYTORCH):
    """
    Common pre-processing for an imagenet trained model.

    - Resize so smallest side is 256
    - Centered crop to 224 x 224
    - Convert image bytes to floating point values in range 0..1
    - [Channels last to channels first (convert to ONNX layout) if model came from pytorch and has NCHW layout]
    - Normalize
      - (value - mean) / stddev
      - for a pytorch model, this applies per-channel normalization parameters
      - for a tensorflow model this simply moves the image bytes into the range -1..1
      - adds a batch dimension with a value of 1
    """

    # These utils cover both cases of typical pytorch/tensorflow pre-processing for an imagenet trained model
    # https://github.com/keras-team/keras/blob/b80dd12da9c0bc3f569eca3455e77762cf2ee8ef/keras/applications/imagenet_utils.py#L177

    steps = [
        Resize(256),
        CenterCrop(224, 224),
        ImageBytesToFloat()
    ]

    if model_source == ModelSource.PYTORCH:
        # pytorch model has NCHW layout
        steps.extend([
            ChannelsLastToChannelsFirst(),
            Normalize([(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)], layout="CHW")
        ])
    else:
        # TF processing involves moving the data into the range -1..1 instead of 0..1.
        # ImageBytesToFloat converts to range 0..1, so we use 0.5 for the mean to move into the range -0.5..0.5
        # and 0.5 for the stddev to expand to -1..1
        steps.append(Normalize([(0.5, 0.5)], layout="HWC"))

    steps.append(Unsqueeze([0]))  # add batch dim

    return steps


def mobilenet(model_file: Path, output_file: Path, model_source: ModelSource = ModelSource.PYTORCH):
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs)

    # support user providing encoded image bytes
    preprocessing = [
        ConvertImageToBGR(),  # custom op to convert jpg/png to BGR (output is HWC)
        ReverseAxis(axis=2, dim_value=3, name="BGR_to_RGB"),
    ]  # Normalization params are for RGB ordering
    # plug in default imagenet pre-processing
    preprocessing.extend(imagenet_preprocessing(model_source))

    pipeline.add_pre_processing(preprocessing)

    # for mobilenet we convert the score to probabilities with softmax if necessary. the TF model includes Softmax
    if model.graph.node[-1].op_type != "Softmax":
        pipeline.add_post_processing([Softmax()])

    new_model = pipeline.run(model)

    onnx.save_model(new_model, str(output_file.resolve()))


def superresolution(model_file: Path, output_file: Path):
    # TODO: There seems to be a split with some super resolution models processing RGB input and some processing
    # the Y channel after converting to YCbCr.
    # For the sake of this example implementation we do the trickier YCbCr processing as that involves joining the
    # Cb and Cr channels with the model output to create the resized image.
    # Model is from https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    # assuming input is *CHW, infer the input sizes from the model.
    # requires the model input and output has a fixed size for the input and output height and width.
    model_input_shape = model.graph.input[0].type.tensor_type.shape
    model_output_shape = model.graph.output[0].type.tensor_type.shape
    assert model_input_shape.dim[-1].HasField("dim_value")
    assert model_input_shape.dim[-2].HasField("dim_value")
    assert model_output_shape.dim[-1].HasField("dim_value")
    assert model_output_shape.dim[-2].HasField("dim_value")

    w_in = model_input_shape.dim[-1].dim_value
    h_in = model_input_shape.dim[-2].dim_value
    h_out = model_output_shape.dim[-2].dim_value
    w_out = model_output_shape.dim[-1].dim_value

    # pre/post processing for https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    pipeline = PrePostProcessor(inputs)
    pipeline.add_pre_processing(
        [
            ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout
            Resize((h_in, w_in)),
            CenterCrop(h_in, w_in),
            # this produces Y, Cb and Cr outputs. each has shape {h_in, w_in}. only Y is input to model
            PixelsToYCbCr(layout="BGR"),
            # if you inserted this Debug step here the 3 outputs from PixelsToYCbCr would also be model outputs
            # Debug(num_inputs=3),
            ImageBytesToFloat(),  # Convert Y to float in range 0..1
            Unsqueeze([0, 1]),  # add batch and channels dim to Y so shape is {1, 1, h_in, w_in}
        ]
    )

    # Post-processing is complicated here. resize the Cb and Cr outputs from the pre-processing to match
    # the model output size, merge those with the Y` model output, and convert back to RGB.

    # create the Steps we need to use in the manual connections
    pipeline.add_post_processing(
        [
            Squeeze([0, 1]),  # remove batch and channels dims from Y'
            FloatToImageBytes(name="Yout_to_bytes"),  # convert Y' to uint8 in range 0..255
            # Resize the Cb values (output 1 from PixelsToYCbCr)
            (
                Resize((h_out, w_out), "HW"),
                [IoMapEntry(producer="PixelsToYCbCr", producer_idx=1, consumer_idx=0)],
            ),
            # the Cb and Cr values are already in the range 0..255 so multiplier is 1. we're using the step to round
            # for accuracy (a direct Cast would just truncate) and clip (to ensure range 0..255) the values post-Resize
            FloatToImageBytes(multiplier=1.0, name="Resized_Cb"),
            (Resize((h_out, w_out), "HW"), [IoMapEntry("PixelsToYCbCr", 2, 0)]),
            FloatToImageBytes(multiplier=1.0, name="Resized_Cr"),
            # as we're selecting outputs from multiple previous steps we need to map them to the inputs using step names
            (
                YCbCrToPixels(layout="BGR"),
                [
                    IoMapEntry("Yout_to_bytes", 0, 0),  # uint8 Y' with shape {h, w}
                    IoMapEntry("Resized_Cb", 0, 1),  # uint8 Cb'
                    IoMapEntry("Resized_Cr", 0, 2),  # uint8 Cr'
                ],
            ),
            ConvertBGRToImage(image_format="jpg"),  # jpg or png are supported
        ]
    )

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add pre and post processing to a model.

        Currently supports updating:
          - super resolution with YCbCr input
          - imagenet trained mobilenet   

        To customize, the logic in the `mobilenet` and `superresolution` functions can be used as a guide.
        Create a pipeline and add the required pre/post processing 'Steps' in the order required. Configure 
        individual steps as needed. 

        The updated model will be written in the same location as the original model, with '.onnx' updated to 
        '.with_pre_post_processing.onnx'
        """,
    )

    parser.add_argument(
        "-t",
        "--model_type",
        type=str,
        required=True,
        choices=["superresolution", "mobilenet"],
        help="Model type.",
    )

    parser.add_argument(
        "--model_source",
        type=str,
        required=False,
        choices=["pytorch", "tensorflow"],
        default="pytorch",
        help="""
        Framework that model came from. In some cases there are known differences that can be taken into account when
        adding the pre/post processing to the model. Currently this equates to choosing different normalization 
        behavior for mobilenet models.
        """,
    )

    parser.add_argument("model", type=Path, help="Provide path to ONNX model to update.")

    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)
    new_model_path = model_path.with_suffix(".with_pre_post_processing.onnx")

    if args.model_type == "mobilenet":
        source = ModelSource.PYTORCH if args.model_source == "pytorch" else ModelSource.TENSORFLOW
        mobilenet(model_path, new_model_path, source)
    else:
        superresolution(model_path, new_model_path)


if __name__ == "__main__":
    main()

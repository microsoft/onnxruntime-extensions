# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import enum
import onnx
import os

from pathlib import Path
from typing import Union
# NOTE: If you're working on this script install onnxruntime_extensions using `pip install -e .` from the repo root
# and run with `python -m onnxruntime_extensions.tools.add_pre_post_processing_to_model`
# Running directly will result in an error from a relative import.
from .pre_post_processing import *


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


def mobilenet(model_file: Path, output_file: Path, model_source: ModelSource, onnx_opset: int = 16):
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs, onnx_opset)

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


def superresolution(model_file: Path, output_file: Path, output_format: str, onnx_opset: int = 16):
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
    pipeline = PrePostProcessor(inputs, onnx_opset)
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
            FloatToImageBytes(name="Y1_uint8"),  # convert Y' to uint8 in range 0..255

            # Resize the Cb values (output 1 from PixelsToYCbCr)
            (Resize((h_out, w_out), "HW"),
             [IoMapEntry(producer="PixelsToYCbCr", producer_idx=1, consumer_idx=0)]),

            # the Cb and Cr values are already in the range 0..255 so multiplier is 1. we're using the step to round
            # for accuracy (a direct Cast would just truncate) and clip (to ensure range 0..255) the values post-Resize
            FloatToImageBytes(multiplier=1.0, name="Cb1_uint8"),

            (Resize((h_out, w_out), "HW"), [IoMapEntry("PixelsToYCbCr", 2, 0)]),
            FloatToImageBytes(multiplier=1.0, name="Cr1_uint8"),

            # as we're selecting outputs from multiple previous steps we need to map them to the inputs using step names
            (
                YCbCrToPixels(layout="BGR"),
                [
                    IoMapEntry("Y1_uint8", 0, 0),  # uint8 Y' with shape {h, w}
                    IoMapEntry("Cb1_uint8", 0, 1),
                    IoMapEntry("Cr1_uint8", 0, 2),
                ],
            ),
            ConvertBGRToImage(image_format=output_format),  # jpg or png are supported
        ]
    )

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


class NLPTaskType(enum.Enum):
    TokenClassification = enum.auto()
    QuestionAnswering = enum.auto()
    SequenceClassification = enum.auto()
    NextSentencePrediction = enum.auto()


class TokenizerType(enum.Enum):
    BertTokenizer = enum.auto()
    SentencePieceTokenizer = enum.auto()


def transformers_and_bert(
    input_model_file: Path,
    output_model_file: Path,
    vocab_file: Path,
    tokenizer_type: Union[TokenizerType, str],
    task_type: Union[NLPTaskType, str],
    onnx_opset: int = 16,
    add_debug_before_postprocessing=False,
):
    """construct the pipeline for a end2end model with pre and post processing. The final model can take text as inputs
    and output the result in text format for model like QA.

    Args:
        input_model_file (Path): the model file needed to be updated.
        output_model_file (Path): where to save the final onnx model.
        vocab_file (Path): the vocab file for the tokenizer.
        task_type (Union[NLPTaskType, str]): the task type of the model.
        onnx_opset (int, optional): the opset version to use. Defaults to 16.
        add_debug_before_postprocessing (bool, optional): whether to add a debug step before post processing. 
            Defaults to False.
    """
    if isinstance(task_type, str):
        task_type = NLPTaskType[task_type]
    if isinstance(tokenizer_type, str):
        tokenizer_type = TokenizerType[tokenizer_type]

    onnx_model = onnx.load(str(input_model_file.resolve(strict=True)))
    # hardcode batch size to 1
    inputs = [create_named_value("input_text", onnx.TensorProto.STRING, [1, "num_sentences"])]

    pipeline = PrePostProcessor(inputs, onnx_opset)
    tokenizer_args = TokenizerParam(
        vocab_or_file=vocab_file,
        do_lower_case=True,
        tweaked_bos_id=0,
        is_sentence_pair=True if task_type in [NLPTaskType.QuestionAnswering,
                                               NLPTaskType.NextSentencePrediction] else False,
    )

    preprocessing = [
        SentencePieceTokenizer(tokenizer_args)
        if tokenizer_type == TokenizerType.SentencePieceTokenizer else BertTokenizer(tokenizer_args),
        # uncomment this line to debug
        # Debug(2),
    ]

    # For verify results with out postprocessing
    postprocessing = [Debug()] if add_debug_before_postprocessing else []
    if task_type == NLPTaskType.QuestionAnswering:
        postprocessing.append((BertTokenizerQADecoder(tokenizer_args), [
            # input_ids
            utils.IoMapEntry("BertTokenizer", producer_idx=0, consumer_idx=2)]))
    elif task_type == NLPTaskType.SequenceClassification:
        postprocessing.append(ArgMax())
    # the other tasks don't need postprocessing or we don't support it yet.

    pipeline.add_pre_processing(preprocessing)
    pipeline.add_post_processing(postprocessing)

    new_model = pipeline.run(onnx_model)
    onnx.save_model(new_model, str(output_model_file.resolve()))


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add pre and post processing to a model.

        Currently supports updating:
        Vision models:
            - super resolution with YCbCr input
            - imagenet trained mobilenet
        NLP models:
        
            - MobileBert with different tasks
            - XLM-Roberta with classification task

        For Vision models:
            To customize, the logic in the `mobilenet` and `superresolution` functions can be used as a guide.
        Create a pipeline and add the required pre/post processing 'Steps' in the order required. Configure 
        individual steps as needed.
        
        For NLP models:
           `transformers_and_bert` can be used for MobileBert QuestionAnswering/Classification tasks,
        or serve as a guide of how to add pre/post processing to a transformer model.
        Usually pre-processing includes adding a tokenizer. Post-processing includes conversion of output_ids to text.
        
        You might need to pass the tokenizer model file (bert vocab file or SentencePieceTokenizer model) 
        and task_type to the function.

        The updated model will be written in the same location as the original model, 
        with '.onnx' updated to '.with_pre_post_processing.onnx'
        """,
    )

    parser.add_argument(
        "-t",
        "--model_type",
        type=str,
        required=True,
        choices=[
            "superresolution",
            "mobilenet",
            "transformers",
        ],
        help="Model type.",
    )

    parser.add_argument(
        "-s",
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

    parser.add_argument(
        "--output_format",
        type=str,
        required=False,
        choices=["jpg", "png"],
        default="png",
        help="Image output format for superresolution model to produce.",
    )

    parser.add_argument(
        "--nlp_task_type",
        type=str,
        choices=["QuestionAnswering",
                 "SequenceClassification",
                 "NextSentencePrediction"],
        required=False,
        help="The downstream task for NLP model.",
    )

    parser.add_argument(
        "--vocab_file",
        type=Path,
        required=False,
        help="Tokenizer model file for BertTokenizer or SentencePieceTokenizer.",
    )

    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["BertTokenizer",
                 "SentencePieceTokenizer"],
        required=False,
        help="Tokenizer model file for BertTokenizer or SentencePieceTokenizer.",
    )

    parser.add_argument(
        "--opset", type=int, required=False, default=16,
        help="ONNX opset to use. Minimum allowed is 16. Opset 18 is required for Resize with anti-aliasing.",
    )

    parser.add_argument("model", type=Path, help="Provide path to ONNX model to update.")

    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)
    new_model_path = model_path.with_suffix(".with_pre_post_processing.onnx")

    if args.model_type == "mobilenet":
        source = ModelSource.PYTORCH if args.model_source == "pytorch" else ModelSource.TENSORFLOW
        mobilenet(model_path, new_model_path, source, args.opset)
    elif args.model_type == "superresolution":
        superresolution(model_path, new_model_path,
                        args.output_format, args.opset)
    else:
        if args.vocab_file is None or args.nlp_task_type is None or args.tokenizer_type is None:
            parser.error("Please provide vocab file/nlp_task_type/tokenizer_type.")
        transformers_and_bert(model_path, new_model_path, args.tokenizer_type, args.vocab_file, args.nlp_task_type)


if __name__ == "__main__":
    main()

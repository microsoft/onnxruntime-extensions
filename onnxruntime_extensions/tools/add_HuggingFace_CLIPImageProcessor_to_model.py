# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
import os
from pathlib import Path
import onnx

from .pre_post_processing import *


class Dict2Class(object):
    '''
    Convert dict to class 
    '''
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def image_processor(args: argparse.Namespace):
    # support user providing encoded image bytes
    steps = [
        ConvertImageToBGR(),  # custom op to convert jpg/png to BGR (output is HWC)
    ]  # Normalization params are for RGB ordering
    if args.do_convert_rgb:
        steps.append(ReverseAxis(axis=2, dim_value=3, name="BGR_to_RGB"))

    if args.do_resize:
        to_size = args.size
        steps.append(Resize(to_size))

    if args.do_center_crop:
        to_size = args.crop_size
        to_size = (to_size, to_size)
        steps.append(CenterCrop(*to_size))

    if args.do_rescale:
        steps.append(ImageBytesToFloat(args.rescale_factor))

    steps.append(ChannelsLastToChannelsFirst())

    if args.do_normalize:
        mean_std = list(zip(args.image_mean, args.image_std))
        layout = 'CHW'
        steps.append(Normalize(mean_std, layout=layout))

    steps.append(Unsqueeze([0]))  # add batch dim

    return steps


def clip_image_processor(model_file: Path, output_file: Path, **kwargs):
    """
    Used for models like stable-diffusion. should be compatible with 
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py

    It's similar to 'CLIP' image processor, and aligns with the HuggingFace class name.

    A typical usage example is used in Stable diffusion.

    :param model_file: The input model file path.
    :param output_file: The output file path, where the finalized model saved to.
    :param Kwargs
        onnx_opset: The opset version of onnx model, default(18).
        do_convert_rgb: Convert image from BGR to RGB. default(True)
        do_resize: Resize the image's (height, width) dimensions to the specified `size`. default(True)
        size: The shortest edge of the image is resized to size. Default(224)
        resample: An optional resampling filter. Default(cubic)
        do_center_crop: Whether to center crop the image to the specified `crop_size`. Default(True)
        crop_size: Size of the output image after applying `center_crop`. Default(224)
        do_rescale: Whether to rescale the image by the specified scale (rescale_factor). Default(True)
        rescale_factor: Scale factor to use if rescaling the image. Default(1/255)
        do_normalize: Whether to normalize the image. Default(True)
        image_mean: Mean values for image normalization. Default([0.485, 0.456, 0.406])
        image_std: Standard deviation values for image normalization. Default([0.229, 0.224, 0.225])


    """
    args = Dict2Class(kwargs)
    # Load model
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs, args.opset)

    preprocessing = image_processor(args)

    pipeline.add_pre_processing(preprocessing)

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))
    print(f"Updated model saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        os.path.basename(__file__),
        description="""Add CLIPImageProcessor to a model.

        The updated model will be written in the same location as the original model, 
        with '.onnx' updated to '.with_clip_processor.onnx'

        Example usage:
            object detection:
            - python -m onnxruntime_extensions.tools.add_HuggingFace_CLIPImageProcessor_to_model model.onnx  
        """,
    )

    parser.add_argument(
        "--opset", type=int, required=False, default=18,
        help="ONNX opset to use. Minimum allowed is 16. Opset 18 is required for Resize with anti-aliasing.",
    )
    
    parser.add_argument(
        "--do_resize", type=bool, required=False, default=True,
        help="Whether to resize the image's (height, width) dimensions to the specified `size`. default(True)",
    )
    parser.add_argument(
        "--size", type=int, required=False, default=224,
        help="The shortest edge of the image is resized to size. Default(224)",
    )
    parser.add_argument(
        "--resample", type=str, default="cubic", choices=["cubic", "nearest","linear"],
        help="Whether to resize the image's (height, width) dimensions to the specified `size`. Default(cubic)",
    )
    parser.add_argument(
        "--do_center_crop", type=bool, default=True,
        help="Whether to center crop the image to the specified `crop_size`. Default(True)",
    )
    parser.add_argument(
        "--crop_size", type=int, default=224,
        help="Size of the output image after applying `center_crop`. Default(224)",
    )
    parser.add_argument(
        "--do_rescale", type=bool, default=True,
        help="Whether to rescale the image by the specified scale (rescale_factor). Default(True)",
    )
    parser.add_argument(
        "--rescale_factor", type=float, default=1/255,
        help="Scale factor to use if rescaling the image. Default(1/255)",
    )
    parser.add_argument(
        "--do_normalize", type=bool, default=True,
        help="Whether to normalize the image. Default(True)",
    )
    parser.add_argument(
        "--image_mean", type=str, default="[0.48145466, 0.4578275, 0.40821073]",
        help=" Mean to use if normalizing the image, default([0.48145466, 0.4578275, 0.40821073])",
    )
    parser.add_argument(
        "--image_std", type=str, default="[0.26862954, 0.26130258, 0.27577711]",
        help="Image standard deviation., default([0.26862954, 0.26130258, 0.27577711]).",
    )
    parser.add_argument(
        "--do_convert_rgb", type=bool, default=True,
        help="Convert image from BGR to RGB. Default(True)",
    )
    parser.add_argument("model", type=Path, help="Provide path to ONNX model to update.")

    args = parser.parse_args()

    args.image_mean = [float(x) for x in args.image_mean.replace('[','').replace(']','').split(",")]
    args.image_std = [float(x) for x in args.image_std.replace('[','').replace(']','').split(",")]  

    model_path = args.model.resolve(strict=True)
    new_model_path = model_path.with_suffix(".with_clip_processor.onnx")

    clip_image_processor(model_path, new_model_path, **vars(args))


if __name__ == "__main__":
    main()

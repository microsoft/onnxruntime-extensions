# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import argparse
from pathlib import Path
import onnx

from .pre_post_processing import *


def image_processor(extend_config: dict):
    # Normalize config key to lower case
    extend_config = {k.lower(): v for k, v in extend_config.items()}

    # support user providing encoded image bytes
    steps = [
        ConvertImageToBGR(),  # custom op to convert jpg/png to BGR (output is HWC)
        ReverseAxis(axis=2, dim_value=3, name="BGR_to_RGB"),
    ]  # Normalization params are for RGB ordering
    if 'resize' in extend_config:
        to_size = extend_config['resize']
        if not isinstance(to_size, int):
            to_size = 256
        steps.append(Resize(to_size))

    if 'center_crop' in extend_config:
        to_size = extend_config['center_crop']
        if not isinstance(to_size, list) or len(to_size) != 2:
            to_size = (224, 224)
        steps.append(CenterCrop(*to_size))

    if 'rescale' in extend_config:
        # default 255
        steps.append(ImageBytesToFloat())

    if 'tochw' in extend_config and extend_config['tochw']:
        steps.append(ChannelsLastToChannelsFirst())

    if 'normalize' in extend_config:
        norm_arg = extend_config['normalize']
        mean_std = [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)]
        layout = 'CHW'
        if isinstance(norm_arg, dict):
            mean_std = norm_arg.get('mean_std', [(0.485, 0.229), (0.456, 0.224), (0.406, 0.225)])
            layout = norm_arg.get('layout', 'CHW')

        steps.append(Normalize(mean_std, layout=layout))

    steps.append(Unsqueeze([0]))  # add batch dim

    return steps


def clip_image_processor(model_file: Path, output_file: Path, onnx_opset: int = 16, extend_config: dict = None):
    """
    Used for models like stable-diffusion. should be compatible with 
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/image_processing_clip.py

    It applies to 'CLIP' image processor, and aligns with the HuggingFace class name.

    A typical usage example is used in Stable diffusion.

    :param model_file: The input model file path.
    :param output_file: The output file path, where the finalized model saved to.
    :param onnx_opset: The opset version of onnx model, default(16).
    :param extend_config: The extended config for the model.
    """
    # Normalize config key to lower case
    if extend_config is not None:
        extend_config = {k.lower(): v for k, v in extend_config.items()}
    else:
        extend_config = {"resize": None, "center_crop": None, "rescale": None, "tochw": True, "normalize": True }
    # Load model
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs, onnx_opset)

    preprocessing = image_processor(extend_config=extend_config)

    pipeline.add_pre_processing(preprocessing)

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))


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
        "--opset", type=int, required=False, default=16,
        help="ONNX opset to use. Minimum allowed is 16. Opset 18 is required for Resize with anti-aliasing.",
    )
    
    parser.add_argument(
        "--extend_config", type=str, required=False, default=None,
        help="extra json config file used for more configurations.",
    )

    parser.add_argument("model", type=Path, help="Provide path to ONNX model to update.")

    args = parser.parse_args()

    model_path = args.model.resolve(strict=True)
    new_model_path = model_path.with_suffix(".with_clip_processor.onnx")

    clip_image_processor(model_path, new_model_path, args.opset, json.loads(
        args.extend_config) if args.extend_config else None)


if __name__ == "__main__":
    main()

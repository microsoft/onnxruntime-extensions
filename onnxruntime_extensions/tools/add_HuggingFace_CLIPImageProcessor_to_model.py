# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
import onnx

from .image_processor import *
from .pre_post_processing import *

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
        extend_config = {"resize": None, "center_crop": None, "normalize": True, "rescale": None}
    # Load model
    model = onnx.load(str(model_file.resolve(strict=True)))
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs, onnx_opset)

    preprocessing = image_processor(extend_config=extend_config)

    pipeline.add_pre_processing(preprocessing)

    new_model = pipeline.run(model)
    onnx.save_model(new_model, str(output_file.resolve()))

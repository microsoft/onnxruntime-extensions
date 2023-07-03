# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from .pre_post_processing import *
import enum

class ModelSource(enum.Enum):
    PYTORCH = 0
    TENSORFLOW = 1
    OTHER = 2


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

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
import onnx
import os
import sys
from pathlib import Path

# add tools dir where pre_post_processing folder is to sys path
script_dir = os.path.dirname(os.path.realpath(__file__))
ort_ext_root = os.path.abspath(os.path.join(script_dir, "..", "..", ".."))
tools_dir = os.path.join(ort_ext_root, "tools")
sys.path.append(tools_dir)

from pre_post_processing import PrePostProcessor
from pre_post_processing.Steps import *
from pre_post_processing.utils import create_named_value, PRE_POST_PROCESSING_ONNX_OPSET


def create_model(output_file: Path):
    """
    Create unit test model. If input is bytes from a jpg we do the following
      - DecodeImage: jpg to BGR
      - EncodeImage: BGR to png (output format is set in the node)
      - DecodeImage: png to BGR

    This is slightly easier to test as we can set the expected output by decoding the original image in the unit test.
    """
    inputs = [create_named_value("image", onnx.TensorProto.UINT8, ["num_bytes"])]

    pipeline = PrePostProcessor(inputs)
    pipeline.add_pre_processing(
        [
            ConvertImageToBGR(),  # jpg/png image to BGR in HWC layout
        ]
    )

    pipeline.add_post_processing(
        [
            ConvertBGRToImage(image_format="png"),  # jpg or png are supported
            ConvertImageToBGR(),  # png to BGR in HWC layout
        ]
    )

    g = onnx.helper.make_graph(
        [
            onnx.helper.make_node("Identity", ["bgr_data_in"], ["bgr_data_out"])
        ],
        "empty",
        [
            onnx.helper.make_tensor_value_info("bgr_data_in", onnx.TensorProto.UINT8, ['h', 'w', 3])
        ],
        [
            onnx.helper.make_tensor_value_info("bgr_data_out", onnx.TensorProto.UINT8, ['h', 'w', 3])
        ]
    )

    onnx_import = onnx.helper.make_operatorsetid('', PRE_POST_PROCESSING_ONNX_OPSET)
    model = onnx.helper.make_model(g, opset_imports=[onnx_import])
    new_model = pipeline.run(model)
    new_model.doc_string = "Model for testing DecodeImage and EncodeImage."
    new_model.graph.doc_string = ""  # clear out all the messages from graph merges
    onnx.save_model(new_model, str(output_file.resolve()))


if __name__ == "__main__":
    create_model(Path('decode_encode_decode_test.onnx'))
